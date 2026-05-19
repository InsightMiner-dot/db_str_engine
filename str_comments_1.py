import os
import asyncio
import pandas as pd
from typing import List, Optional
from pydantic import Field, create_model
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm.asyncio import tqdm_asyncio  # Async Progress bar

# Force override system variables with .env variables
load_dotenv(override=True)

# ==========================================
# Phase 1: DYNAMIC SCHEMA GENERATOR
# ==========================================
def build_dynamic_schema(target_columns: list, mapping_df: pd.DataFrame):
    fields = {}
    mapping_cols_lower = {str(c).strip().lower(): c for c in mapping_df.columns}
    
    for col in target_columns:
        safe_name = col.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        
        if col.strip().lower() in mapping_cols_lower:
            original_map_col = mapping_cols_lower[col.strip().lower()]
            desc = f"Map strictly to one of the exact allowed values for '{original_map_col}'."
        else:
            desc = f"Extract or infer the correct value for '{col}' based on the text context."
            
        fields[safe_name] = (Optional[str], Field(None, description=desc))

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
    async with semaphore:
        extracted_rows = []
        try:
            response = await parsing_chain.ainvoke({
                "allowed_values_context": allowed_values_context,
                "comment": str(raw_comment)
            })
            
            if response and response.items:
                for item in response.items:
                    row_dict = item.model_dump()
                    final_row = base_row_data.copy()
                    
                    for original_col, safe_key in zip(target_columns, pydantic_field_names):
                        final_row[original_col] = row_dict[safe_key]
                        
                    # --- OH/LC FALLBACK LOGIC ---
                    oh_lc_col = next((c for c in target_columns if 'oh' in str(c).lower() and 'lc' in str(c).lower()), None)
                    costcat_col = next((c for c in target_columns if 'costcat' in str(c).lower() or 'cost cat' in str(c).lower()), None)
                    
                    if oh_lc_col and costcat_col:
                        current_oh_lc = final_row.get(oh_lc_col)
                        
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
                    # --- END OH/LC FALLBACK LOGIC ---
                        
                    extracted_rows.append(final_row)
            else:
                fallback_row = base_row_data.copy()
                for col in target_columns:
                    fallback_row[col] = None
                extracted_rows.append(fallback_row)
                
        except Exception as e:
            error_row = base_row_data.copy()
            for col in target_columns:
                error_row[col] = None
                
            context_col_match = [c for c in target_columns if 'context' in c.lower() or 'driver' in c.lower()]
            if context_col_match:
                error_row[context_col_match[0]] = f"SYSTEM ERROR: {str(e)}"
                
            extracted_rows.append(error_row)
            
        return extracted_rows

# ==========================================
# Phase 3: ORCHESTRATOR & FAIL-SAFE LOGIC
# ==========================================
async def process_dynamic_comments_async(input_file: str, mapping_file: str, output_file: str, max_concurrency: int = 3):
    print("Loading source files...")
    
    mapping_df = pd.read_excel(mapping_file)
    input_df = pd.read_excel(input_file)
    
    comment_col = [c for c in input_df.columns if 'omment' in c.lower()][0] 
    comment_idx = input_df.columns.get_loc(comment_col)
    
    base_columns = input_df.columns[:comment_idx + 1].tolist()
    target_columns = input_df.columns[comment_idx + 1:].tolist()
    
    print("Generating schema from blank target columns...")
    DynamicPayloadSchema, pydantic_field_names = build_dynamic_schema(target_columns, mapping_df)
    
    allowed_values_context = ""
    for col in mapping_df.columns:
        unique_vals = [str(x) for x in mapping_df[col].dropna().unique()]
        allowed_values_context += f"- {col}: {', '.join(unique_vals)}\n"

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([azure_endpoint, azure_api_key, azure_deployment, azure_api_version]):
        raise ValueError("Missing Azure OpenAI credentials. Please verify your .env file.")

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        temperature=0.0,
        max_retries=5 
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

    print(f"\nQueueing {len(input_df)} rows for concurrent extraction...")
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
        
        await asyncio.sleep(0.05) 

    # ==========================================
    # FAULT-TOLERANT EXECUTION BLOCK
    # ==========================================
    all_structured_rows = []
    print(f"\nExecuting API calls with a concurrency limit of {max_concurrency}...")
    
    try:
        # Using as_completed allows us to process results the millisecond they finish
        for completed_task in tqdm_asyncio.as_completed(tasks, desc="Processing Variance Comments"):
            result = await completed_task
            all_structured_rows.extend(result)
            
    except (Exception, KeyboardInterrupt) as e:
        # This catches Manual Stops (Ctrl+C) or catastrophic system failures
        print(f"\n[INTERRUPT DETECTED]: {str(e)}")
        print("Halting extraction and saving all rows completed up to this point...")

    # ==========================================
    # Phase 4: EXPORT DATA
    # ==========================================
    if len(all_structured_rows) > 0:
        output_df = pd.DataFrame(all_structured_rows)
        output_df = output_df[input_df.columns]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # If it was interrupted, append "_partial" to the filename so you know it didn't finish
        if len(all_structured_rows) < len(tasks):
            file_name, file_ext = os.path.splitext(output_file)
            output_file = f"{file_name}_partial{file_ext}"
            
        output_df.to_excel(output_file, index=False)
        print(f"\nSuccess! Processed {len(all_structured_rows)} structured rows. File written to: {output_file}")
    else:
        print("\nNo rows were successfully completed. Nothing to save.")

# ==========================================
# GATEWAY
# ==========================================
if __name__ == "__main__":
    asyncio.run(
        process_dynamic_comments_async(
            input_file="data/input_comments.xlsx",
            mapping_file="data/mapping_file.xlsx",
            output_file="data/structured_variance_output.xlsx",
            max_concurrency=3 
        )
    )
