import os
import pandas as pd
from typing import List, Optional
from pydantic import Field, create_model
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==========================================
# 1. DYNAMIC SCHEMA GENERATOR
# ==========================================
def build_dynamic_schema(mapping_df: pd.DataFrame):
    """
    Reads the mapping file headers and generates a Pydantic model at runtime.
    Automatically appends the required fixed columns.
    """
    fields = {}
    
    # 1. Dynamically add every column from the mapping file
    for col in mapping_df.columns:
        # Create a python-safe attribute name (e.g., 'OH/LC' -> 'oh_lc')
        safe_name = col.strip().lower().replace(" ", "_").replace("/", "_")
        fields[safe_name] = (Optional[str], Field(
            None, 
            description=f"Map this to one of the exact unique values provided for '{col}'."
        ))

    # 2. Append your fixed additional columns
    fields["scenario_a_total"] = (Optional[str], Field(None, description="Baseline, budget, or previous year total value."))
    fields["scenario_b_total"] = (Optional[str], Field(None, description="Actual, revised, or current forecast total value."))
    fields["variance_amount"] = (Optional[str], Field(None, description="The isolated variance amount with units (e.g., '-7M')."))
    fields["context_driver"] = (Optional[str], Field(None, description="The underlying logic, driver, or business reason behind the variance."))
    fields["region_for_variance"] = (Optional[str], Field(None, description="Specific country or region tied directly to this variance chunk."))

    # 3. Create the models at runtime
    DynamicRowModel = create_model('DynamicRowModel', **fields)
    DynamicPayloadModel = create_model('DynamicPayloadModel', items=(List[DynamicRowModel], Field(..., description="List of extracted lines.")))
    
    return DynamicPayloadModel, list(fields.keys())

# ==========================================
# 2. RUNTIME EXTRACTION PIPELINE
# ==========================================
def process_dynamic_comments(input_file: str, mapping_file: str, output_file: str):
    print("Loading source files...")
    
    # Read mapping file
    mapping_df = pd.read_excel(mapping_file)
    mapping_columns = mapping_df.columns.tolist()
    
    # Read input file (handles typos like 'Commnets' vs 'Comments')
    input_df = pd.read_excel(input_file)
    comment_col = [c for c in input_df.columns if 'omment' in c.lower()][0] 
    
    # Generate the Pydantic Schema dynamically
    print("Generating dynamic LLM schema based on mapping file...")
    DynamicPayloadSchema, pydantic_field_names = build_dynamic_schema(mapping_df)
    
    # Build a dictionary of allowed values for the Prompt
    allowed_values_context = ""
    for col in mapping_columns:
        unique_vals = [str(x) for x in mapping_df[col].dropna().unique()]
        allowed_values_context += f"- {col}: {', '.join(unique_vals)}\n"

    # Setup Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",      
        api_version="2024-08-01-preview",     
        temperature=0.0                       
    )
    
    # Bind the dynamically created schema
    structured_llm = llm.with_structured_output(DynamicPayloadSchema)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Financial Controller building a structured database from raw variance comments.\n"
            "You must extract financial impacts and map them to our internal structural dimensions.\n\n"
            
            "--- STRICT MAPPING RULES ---\n"
            "For the dimensional columns, you MUST choose from these allowed values whenever possible:\n"
            "{allowed_values_context}\n\n"
            
            "--- SHORTHAND TRANSLATION ---\n"
            "- 'o/w' = 'of which'. Split these nested elements out into their own individual rows!\n"
            "- 'M' = Millions (e.g., '-7M').\n"
        )),
        ("user", "Parse this variance comment text into structured rows:\n\n{comment}")
    ])

    parsing_chain = prompt_template | structured_llm

    all_structured_rows = []

    print(f"Executing dynamic extraction across {len(input_df)} rows...")
    for idx, row in input_df.iterrows():
        raw_comment = row[comment_col]
        
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            continue
            
        try:
            response = parsing_chain.invoke({
                "allowed_values_context": allowed_values_context,
                "comment": str(raw_comment)
            })
            
            if response and response.items:
                for item in response.items:
                    # Convert the dynamic Pydantic model into a dictionary
                    row_dict = item.model_dump()
                    
                    # We also want to keep the original text for auditing
                    row_dict["Original_Comment"] = raw_comment
                    all_structured_rows.append(row_dict)
                    
        except Exception as e:
            print(f" [ALERT] Failed on row {idx}: {e}")
            # Create an empty dict matching our dynamic schema keys for error tracking
            error_dict = {key: None for key in pydantic_field_names}
            error_dict["Original_Comment"] = raw_comment
            error_dict["context_driver"] = f"SYSTEM ERROR: {str(e)}"
            all_structured_rows.append(error_dict)

    # ==========================================
    # 3. BUILD DATAFRAME & EXPORT
    # ==========================================
    output_df = pd.DataFrame(all_structured_rows)
    
    # Clean up column names for the final Excel file (revert snake_case to readable names)
    rename_map = {safe_key: orig_col for safe_key, orig_col in zip(pydantic_field_names[:len(mapping_columns)], mapping_columns)}
    rename_map.update({
        "scenario_a_total": "Scenario A total",
        "scenario_b_total": "Scenario B total",
        "variance_amount": "Variance amount",
        "context_driver": "Context / Driver",
        "region_for_variance": "Region for variance",
        "Original_Comment": "Original comments"
    })
    
    output_df = output_df.rename(columns=rename_map)
    output_df.to_excel(output_file, index=False)
    print(f"\nSuccess! Dynamic file written to: {output_file}")


if __name__ == "__main__":
    process_dynamic_comments(
        input_file="input_comments.xlsx",
        mapping_file="mapping_file.xlsx",
        output_file="structured_variance_output.xlsx"
    )
