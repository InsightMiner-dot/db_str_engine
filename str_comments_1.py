import os
import pandas as pd
from typing import List, Optional
from pydantic import Field, create_model
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Force override system variables with .env variables
load_dotenv(override=True)

# ==========================================
# 1. DYNAMIC SCHEMA GENERATOR
# ==========================================
def build_dynamic_schema(target_columns: list, mapping_df: pd.DataFrame):
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

    DynamicRowModel = create_model('DynamicRowModel', **fields)
    DynamicPayloadModel = create_model(
        'DynamicPayloadModel', 
        items=(List[DynamicRowModel], Field(..., description="List of all extracted variance line items."))
    )
    
    return DynamicPayloadModel, list(fields.keys())

# ==========================================
# 2. RUNTIME EXTRACTION PIPELINE
# ==========================================
def process_dynamic_comments(input_file: str, mapping_file: str, output_file: str):
    print("Loading source files...")
    
    mapping_df = pd.read_excel(mapping_file)
    input_df = pd.read_excel(input_file)
    
    # Dynamically locate the 'Comments' column
    comment_col = [c for c in input_df.columns if 'omment' in c.lower()][0] 
    comment_idx = input_df.columns.get_loc(comment_col)
    
    # Split the columns based on your schema layout:
    # Base columns (File_name, Category, etc.) are everything up to and including 'Comments'
    base_columns = input_df.columns[:comment_idx + 1].tolist()
    
    # Target columns (Region, Market, OH_LC, etc.) are everything after 'Comments'
    target_columns = input_df.columns[comment_idx + 1:].tolist()
    
    print("Generating schema from blank target columns...")
    DynamicPayloadSchema, pydantic_field_names = build_dynamic_schema(target_columns, mapping_df)
    
    # Build a dictionary of allowed values for the Prompt
    allowed_values_context = ""
    for col in mapping_df.columns:
        unique_vals = [str(x) for x in mapping_df[col].dropna().unique()]
        allowed_values_context += f"- {col}: {', '.join(unique_vals)}\n"

    # Load Azure config
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
        temperature=0.0 
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
        )),
        ("user", "Parse this variance comment text into structured rows:\n\n{comment}")
    ])

    parsing_chain = prompt_template | structured_llm

    all_structured_rows = []

    print(f"Executing extraction across {len(input_df)} rows...")
    for idx, row in input_df.iterrows():
        raw_comment = row[comment_col]
        
        # Skip if there is no comment to parse
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            continue
            
        # 1. Capture the base metadata (File_name, Category, Scenarios, Year, Month, Comments)
        base_row_data = {col: row[col] for col in base_columns}
            
        try:
            # 2. Ask LLM to extract just the target columns
            response = parsing_chain.invoke({
                "allowed_values_context": allowed_values_context,
                "comment": str(raw_comment)
            })
            
            if response and response.items:
                # 3. For every item the LLM found, duplicate the base data and append the LLM data
                for item in response.items:
                    row_dict = item.model_dump()
                    
                    # Start with a fresh copy of the base metadata
                    final_row = base_row_data.copy()
                    
                    # Map the LLM's target columns back to the exact input file headers
                    for original_col, safe_key in zip(target_columns, pydantic_field_names):
                        final_row[original_col] = row_dict[safe_key]
                        
                    all_structured_rows.append(final_row)
            else:
                # Fallback if no items extracted
                fallback_row = base_row_data.copy()
                for col in target_columns:
                    fallback_row[col] = None
                all_structured_rows.append(fallback_row)
                    
        except Exception as e:
            print(f" [ALERT] Failed on row {idx}: {e}")
            error_row = base_row_data.copy()
            for col in target_columns:
                error_row[col] = None
                
            # Log error in the context column if it exists
            context_col_match = [c for c in target_columns if 'context' in c.lower() or 'driver' in c.lower()]
            if context_col_match:
                error_row[context_col_match[0]] = f"SYSTEM ERROR: {str(e)}"
                
            all_structured_rows.append(error_row)

    # ==========================================
    # 3. BUILD DATAFRAME & EXPORT
    # ==========================================
    output_df = pd.DataFrame(all_structured_rows)
    
    # Ensure columns match the exact order of the original input file
    output_df = output_df[input_df.columns]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_excel(output_file, index=False)
    print(f"\nSuccess! Structured file written to: {output_file}")

if __name__ == "__main__":
    process_dynamic_comments(
        input_file="data/input_comments.xlsx",
        mapping_file="data/mapping_file.xlsx",
        output_file="data/structured_variance_output.xlsx"
    )
