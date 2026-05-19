import os
import asyncio
import pandas as pd
from typing import List, Optional
from pydantic import Field, create_model
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables (Overrides system env to prioritize the .env file)
load_dotenv(override=True)

# ==========================================
# Phase 1: DYNAMIC SCHEMA GENERATOR
# ==========================================
def build_dynamic_schema(target_columns: list, mapping_df: pd.DataFrame):
    """
    Dynamically builds the Pydantic schema based on the blank columns in the input file.
    If a column exists in the mapping file, it enforces strict selection.
    """
    fields = {}
    mapping_cols_lower = {str(c).strip().lower(): c for c in mapping_df.columns}
    
    for col in target_columns:
        safe_name = col.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        
        # Enforce strict mapping if the column exists in the mapping file
        if col.strip().lower() in mapping_cols_lower:
            original_map_col = mapping_cols_lower[col.strip().lower()]
            desc = f"Map strictly to one of the exact allowed values for '{original_map_col}'."
        else:
            desc = f"Extract or infer the correct value for '{col}' based on the text context."
            
        fields[safe_name] = (Optional[str], Field(None, description=desc))

    # Create the row model and the wrapper payload model at runtime
    DynamicRowModel = create_model('DynamicRowModel', **fields)
    DynamicPayloadModel = create_model(
        'DynamicPayloadModel', 
        items=(List[DynamicRowModel], Field(..., description="List of all extracted variance line items."))
    )
    
    return DynamicPayloadModel, list(fields.keys())

# ==========================================
# Phase 2: ASYNC WORKER (Processes 1 Row)
# ==========================================
async def async_process_single_row(
    idx, raw_comment, base_row_data, target_columns, pydantic_field_names, 
    allowed_values_context, parsing_chain, semaphore
):
    """
    Processes a single row asynchronously.
    Protected by a semaphore to prevent Azure Rate Limit errors.
    """
    async with semaphore:
        extracted_rows = []
        
        try:
            # Trigger the LLM chain asynchronously
            response = await parsing_chain.ainvoke({
                "allowed_values_context": allowed_values_context,
                "comment": str(raw_comment)
            })
            
            # If the LLM successfully extracted data
            if response and response.items:
                for item in response.items:
                    row_dict = item.model_dump()
                    
                    # Start with a fresh copy of the base metadata (File_name, Category, etc.)
                    final_row = base_row_data.copy()
                    
                    # Map the LLM's target columns back to the exact input file headers
                    for original_col, safe_key in zip(target_columns, pydantic_field_names):
                        final_row[original_col] = row_dict[safe_key]
                        
                    # ----------------------------------------------------
                    # CUSTOM BUSINESS LOGIC: OH / LC FALLBACK
                    # ----------------------------------------------------
                    # Dynamically find the OH_LC and CostCat columns
                    oh_lc_col = next((c for c in target_columns if 'oh' in str(c).lower() and 'lc' in str(c).lower()), None)
                    costcat_col = next((c for c in target_columns if 'costcat' in str(c).lower() or 'cost cat' in str(c).lower()), None)
                    
                    if oh_lc_col and costcat_col:
                        current_oh_lc = final_row.get(oh_lc_col)
                        
                        # If LLM missed OH/LC, infer it from CostCat_desc
                        if pd.isna(current_oh_lc) or str(current_oh_lc).strip() == "" or str(current_oh_lc).lower() == "none":
                            cost_cat_val = str(final_row.get(costcat_col, "")).strip().upper()
                            
                            lc_categories = ["FSA COSTS", "PERSONAL COSTS", "CONTRACTORS"]
                            oh_categories = [
                                "PROCURED SERVICES", "TRAVEL & MEALS", "EMPLOYEE WELFARE", 
                                "OPERATING COSTST", "OPERATING COSTS", "EMPLOYEE ACTIVITY COST", 
                                "RECHARGE", "OFFICE SPACE", "COMPANY CAR COSTS", 
                                "RECHARGE OUTSIDFE", "RECHARGE OUTSIDE", "TAX", 
                                "PROVISION DEBT", "DEPRECIATION", "FUNCTIONAL TASK"
                            ]
                            
                            if any(lc_item in cost_cat_val for lc_item in lc_categories):
                                final_row[oh_lc_col] = "LC"
                            elif any(oh_item in cost_cat_val for oh_item in oh_categories):
                                final_row[oh_lc_col] = "OH"
                    # ----------------------------------------------------
                        
                    extracted_rows.append(final_row)
            else:
                # Fallback: keep base metadata, but target columns remain blank
                fallback_row = base_row_data.copy()
                for col in target_columns:
                    fallback_row[col] = None
                extracted_rows.append(fallback_row)
                
        except Exception as e:
            # Fallback for API failures or severe parsing errors
            print(f" [ALERT] Failed on row {idx}: {e}")
            error_row = base_row_data.copy()
            for col in target_columns:
                error_row[col] = None
                
            # Log the exception directly into the Context/Driver column if it exists
            context_col_match = [c for c in target_columns if 'context' in c.lower() or 'driver' in c.lower()]
            if context_col_match:
                error_row[context_col_match[0]] = f"SYSTEM ERROR: {str(e)}"
                
            extracted_rows.append(error_row)
            
        return extracted_rows

# ==========================================
# Phase 3: ORCHESTRATOR
# ==========================================
async def process_dynamic_comments_async(input_file: str, mapping_file: str, output_file: str, max_concurrency: int = 3):
    print("Loading source files...")
    
    mapping_df = pd.read_excel(mapping_file)
    input_df = pd.read_excel(input_file)
    
    # Identify target column splitting points
    comment_col = [c for c in input_df.columns if 'omment' in c.lower()][0] 
    comment_idx = input_df.columns.get_loc(comment_col)
    
    base_columns = input_df.columns[:comment_idx + 1].tolist()
    target_columns = input_df.columns[comment_idx + 1:].tolist()
    
    print("Generating schema from blank target columns...")
    DynamicPayloadSchema, pydantic_field_names = build_dynamic_schema(target_columns, mapping_df)
    
    # Inject mapped unique values into the system prompt
    allowed_values_context = ""
    for col in mapping_df.columns:
        unique_vals = [str(x) for x in mapping_df[col].dropna().unique()]
        allowed_values_context += f"- {col}: {', '.join(unique_vals)}\n"

    # Azure Credentials Check
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([azure_endpoint, azure_api_key, azure_deployment, azure_api_version]):
        raise ValueError("Missing Azure OpenAI credentials. Please verify your .env file.")

    # Instantiate LLM with Retry Logic for Rate Limiting
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        temperature=0.0,
        max_retries=5  # Resilient to rate limits
    )
    
    structured_llm = llm.with_structured_output(DynamicPayloadSchema)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Financial Controller building a structured database from raw variance comments.\n"
            "Your job is to read the comment and fill out the required data columns.\n\n"
            
            "--- STRICT MAPPING RULES ---\n"
            "For organizational columns, you MUST choose from these allowed values whenever possible:\n"
            "{allowed_values_context}\n\n"
            
            "--- SHORTHAND TRANSLATION ---\n"
            "- 'o/w' = 'of which'. Split nested elements out into their own individual rows!\n"
            "- 'M' = Millions (e.g., '-7M').\n"
            "- 'OH' = Over Head Cost, 'LC' = Labour Cost.\n"
        )),
        ("user", "Parse this variance comment text into structured rows:\n\n{comment}")
    ])

    parsing_chain = prompt_template | structured_llm
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    print(f"Queueing {len(input_df)} rows for concurrent extraction...")
    for idx, row in input_df.iterrows():
        raw_comment = row[comment_col]
        
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            continue
            
        base_row_data = {col: row[col] for col in base_columns}
        
        task = async_process_single_row(
            idx, raw_comment, base_row_data, target_columns, pydantic_field_names, 
            allowed_values_context, parsing_chain, semaphore
        )
        tasks.append(task)
        
        # Micro-delay to stagger API calls and avoid bursting Azure limits
        await asyncio.sleep(0.1) 

    # Execute all tasks concurrently
    print(f"Executing API calls with a concurrency limit of {max_concurrency}...")
    results = await asyncio.gather(*tasks)
    
    # Flatten the multi-dimensional results array
    all_structured_rows = [item for sublist in results for item in sublist]

    # ==========================================
    # Phase 4: EXPORT DATA
    # ==========================================
    output_df = pd.DataFrame(all_structured_rows)
    
    # Ensure columns match the original layout exactly
    output_df = output_df[input_df.columns]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_excel(output_file, index=False)
    print(f"\nSuccess! Async pipeline complete. Structured file written to: {output_file}")

# ==========================================
# GATEWAY
# ==========================================
if __name__ == "__main__":
    asyncio.run(
        process_dynamic_comments_async(
            input_file="data/input_comments.xlsx",
            mapping_file="data/mapping_file.xlsx",
            output_file="data/structured_variance_output.xlsx",
            max_concurrency=3  # Keeps API requests under the speed limit
        )
    )
