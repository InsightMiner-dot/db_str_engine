import os
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv  # <-- 1. Import dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 2. Load environment variables from the .env file
load_dotenv()

# ==========================================
# 1. DEFINE YOUR EXACT TARGET SCHEMA
# ==========================================
class StructuredVarianceRow(BaseModel):
    region: Optional[str] = Field(None, description="The geographical region or theater, e.g., 'Europe', 'APAC', 'AMER'.")
    oh_lc: Optional[str] = Field(None, description="Overhead Local Currency indicator or identifier if mentioned.")
    division_desc: Optional[str] = Field(None, description="The division description mapped or extracted from the text.")
    function_desc: Optional[str] = Field(None, description="The corporate function description (e.g., Finance, HR, IT, Supply Chain).")
    department_desc: Optional[str] = Field(None, description="The granular department description.")
    entity_desc: Optional[str] = Field(None, description="The legal entity name or identifier, e.g., 'NMUK'.")
    costcat_description: Optional[str] = Field(None, description="The standardized Cost Category description. MUST align with the unique categories in your master database if possible.")
    scenario_a_total: Optional[str] = Field(None, description="The baseline, budget, or previous year total value.")
    scenario_b_total: Optional[str] = Field(None, description="The actual, revised, or current forecast total value.")
    variance_amount: Optional[str] = Field(None, description="The isolated variance amount with its units (e.g., '-7M', '-33 M', '-2.5M').")
    context_driver: Optional[str] = Field(None, description="The underlying logic, driver, business reason, or background behind this specific variance.")
    region_for_variance: Optional[str] = Field(None, description="Specific country or regional assignment tied directly to this variance chunk.")

class ParsedCommentPayload(BaseModel):
    items: List[StructuredVarianceRow] = Field(..., description="List of all separate structural lines extracted out of the target comment.")

# ==========================================
# 2. RUNTIME EXTRACTION PIPELINE
# ==========================================
def process_variance_comments(comments_file_path: str, master_file_path: str, output_file_path: str):
    
    print("Connecting to Azure OpenAI Model Endpoint...")
    # LangChain automatically picks up AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT from the environment
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",      
        api_version="2024-08-01-preview",     
        temperature=0.0                       
    )
    
    structured_llm = llm.with_structured_output(ParsedCommentPayload)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Senior Financial Controller and systems data migration specialist.\n"
            "Your job is to parse highly compressed, shorthand variance comment entries and extract "
            "individual structural database rows out of them.\n\n"
            
            "CRITICAL CATEGORY VALIDATION:\n"
            "You must map the extracted 'costcat_description' field to match one of these unique master database categories as closely as possible:\n"
            "--- MASTER DATABASE UNIQUE CATEGORIES ---\n"
            "{allowed_categories}\n"
            "--- END MASTER DATABASE UNIQUE CATEGORIES ---\n\n"
            
            "SHORTHAND TRANSLATION LOGIC:\n"
            "- 'o/w' or 'ow' = 'of which'. Split these nested elements out into their own individual rows.\n"
            "- 'M' = Millions (e.g., translate or capture '-7M', '-2.5M').\n"
        )),
        ("user", "Parse this variance comment text into individual database items:\n\n{comment}")
    ])

    parsing_chain = prompt_template | structured_llm

    print(f"Reading target files:\n - Master File: {master_file_path}\n - Comments File: {comments_file_path}")
    
    master_df = pd.read_excel(master_file_path)
    allowed_categories_list = master_df["Cost Category"].dropna().unique().tolist()
    allowed_categories_str = "\n".join(allowed_categories_list)

    comments_df = pd.read_excel(comments_file_path)
    all_structured_rows = []

    print("\nExecuting model inference loops across data rows...")
    for idx, row in comments_df.iterrows():
        raw_comment = row['Comment']
        
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            continue
            
        print(f" -> Structuring Row Index {idx}...")
        
        try:
            response = parsing_chain.invoke({
                "allowed_categories": allowed_categories_str,
                "comment": str(raw_comment)
            })
            
            if response and response.items:
                for item in response.items:
                    new_record = {
                        "Region": item.region,
                        "OH_LC": item.oh_lc,
                        "Division_Desc": item.division_desc,
                        "Function_Desc": item.function_desc,
                        "Department_Desc": item.department_desc,
                        "Entity_Desc": item.entity_desc,
                        "CostCat Description": item.costcat_description,
                        "Scenario A total": item.scenario_a_total,
                        "Scenario B total": item.scenario_b_total,
                        "Variance amount": item.variance_amount,
                        "Original comments": raw_comment,  
                        "Context / Driver": item.context_driver,
                        "region for variance": item.region_for_variance
                    }
                    all_structured_rows.append(new_record)
            else:
                all_structured_rows.append({
                    "Region": None, "OH_LC": None, "Division_Desc": None, "Function_Desc": None, "Department_Desc": None,
                    "Entity_Desc": None, "CostCat Description": "Unparsed", "Scenario A total": None, "Scenario B total": None,
                    "Variance amount": None, "Original comments": raw_comment, "Context / Driver": "LLM skipped row classification",
                    "region for variance": None
                })
                
        except Exception as e:
            print(f"    [ALERT] System anomaly caught on row {idx}: {e}")
            all_structured_rows.append({
                "Region": None, "OH_LC": None, "Division_Desc": None, "Function_Desc": None, "Department_Desc": None,
                "Entity_Desc": None, "CostCat Description": "System Error", "Scenario A total": None, "Scenario B total": None,
                "Variance amount": None, "Original comments": raw_comment, "Context / Driver": f"Exception Trace: {str(e)}",
                "region for variance": None
            })

    output_df = pd.DataFrame(all_structured_rows)
    
    column_order = [
        "Region", "OH_LC", "Division_Desc", "Function_Desc", "Department_Desc", 
        "Entity_Desc", "CostCat Description", "Scenario A total", "Scenario B total", 
        "Variance amount", "Original comments", "Context / Driver", "region for variance"
    ]
    output_df = output_df[column_order]
    
    output_df.to_excel(output_file_path, index=False)
    print(f"\nProcessing Pipeline Successfully Wrapped! File written to: {output_file_path}")

if __name__ == "__main__":
    process_variance_comments(
        comments_file_path="variance_comments.xlsx",
        master_file_path="master_categories.xlsx",
        output_file_path="structured_variance_output.xlsx"
    )