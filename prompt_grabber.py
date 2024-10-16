import os
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager
from ibm_watson_machine_learning.foundation_models.utils.enums import PromptTemplateFormats
import pandas as pd

def get_latest_prompt_template(promptType):
    prompt_mgr = PromptTemplateManager(
        credentials={
            "apikey": os.environ.get("IBM_CLOUD_API_KEY"),
            "url": os.environ.get("WX_URL"),
        },
        space_id=os.environ.get("WX_SPACE_ID")
    )
    
    df_prompts = prompt_mgr.list()

    df_prompts = df_prompts.assign(
            NAME=df_prompts['NAME'].astype(str),
            LAST_MODIFIED=pd.to_datetime(df_prompts['LAST MODIFIED'])
        )

    filtered_df = df_prompts[df_prompts['NAME'] == promptType]

    if filtered_df.empty:
        raise ValueError(f"Prompt file does not exist for NAME = {promptType}")

    # Find the latest record and prompt id based on 'LAST MODIFIED'
    latest_index = filtered_df['LAST MODIFIED'].idxmax()
    latest_record = filtered_df.loc[latest_index]

    latest_prompt_id = latest_record['ID']

    # Load the prompt template using the latest ID and format type as string
    loaded_prompt_template_string = prompt_mgr.load_prompt(latest_prompt_id, PromptTemplateFormats.STRING)
    
    return loaded_prompt_template_string

def update_prompt_templates(prompt_types):
    for prompt_type in prompt_types:
        with open(f"prompt_templates/{prompt_type}", "w") as f:
            f.write(get_latest_prompt_template(prompt_type))