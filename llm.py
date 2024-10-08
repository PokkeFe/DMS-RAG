import os

# wx.ai
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager
from ibm_watson_machine_learning.foundation_models.utils.enums import PromptTemplateFormats

# langchain
from langchain_core import tools
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from langchain_ibm import WatsonxLLM, ChatWatsonx

from langchain_community.utilities import SQLDatabase

def query(input: str) -> str:

    print 

    MODEL_ID = 'ibm/granite-13b-chat-v2'
    parameters = {  
        GenTextParamsMetaNames.DECODING_METHOD: "sample",  
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,  
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,  
        GenTextParamsMetaNames.TEMPERATURE: 0.5,  
        GenTextParamsMetaNames.TOP_K: 50,  
        GenTextParamsMetaNames.TOP_P: 1,  
    }  
    from langchain_ibm import ChatWatsonx  
    watsonx_llm = ChatWatsonx(  
    model_id="meta-llama/llama-3-70b-instruct",  
    url="https://us-south.ml.cloud.ibm.com",  
    apikey=os.environ.get("IBM_CLOUD_API_KEY"),  
    project_id=os.environ.get("WX_PROJECT_ID"),  
    params=parameters,  
    )

    template = PromptTemplate.from_template("Generate an example SQL query that would accomplish the following task:\n{topic}")

    key = os.environ.get("IBM_CLOUD_API_KEY")

    prompt_value = template.invoke({"topic": input})

    print("Stream")
    for chunk in watsonx_llm.stream(prompt_value):
        print(chunk.content, end="")
    print()
    print("Stream End")

    llm_chain = template | watsonx_llm


    return llm_chain.invoke(prompt_value)