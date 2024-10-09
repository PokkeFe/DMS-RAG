import os

from typing import Annotated, Literal
from typing_extensions import TypedDict

# wx.ai
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager
from ibm_watson_machine_learning.foundation_models.utils.enums import PromptTemplateFormats

# langchain
from langchain import hub
from langchain_core import tools
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_ibm import WatsonxLLM, ChatWatsonx

from langchain_community.utilities import SQLDatabase

import duckduckgo_search

# langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# pydantic
from pydantic import BaseModel, Field

from db2_loader import get_db2_database

# State definitions
class State(TypedDict):
    user_input: str
    graph_output: str


def classify_node(state: State) -> State:
    return {}

def sqlgen_node(state: State) -> State:
    return {}

def sqlexec_node(state: State) -> State:
    return {}

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def query(q: str) -> str:

    print 

    MODEL_ID = 'ibm/granite-13b-chat-v2'
    parameters = {  
        GenTextParamsMetaNames.DECODING_METHOD: "sample",  
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,  
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,  
        GenTextParamsMetaNames.TEMPERATURE: 0.5,  
        GenTextParamsMetaNames.TOP_K: 50,  
        GenTextParamsMetaNames.TOP_P: 1,
        GenTextParamsMetaNames.STOP_SEQUENCES: ["\n"]
    }   
    watsonx_llm = WatsonxLLM(  
    model_id="meta-llama/llama-3-2-90b-vision-instruct",  
    url= "https://us-south.ml.cloud.ibm.com",  
    apikey= os.environ.get("IBM_CLOUD_API_KEY"), 
    project_id=os.environ.get("WX_PROJECT_ID"),  
    params=parameters,  
    )
    key = os.environ.get("IBM_CLOUD_API_KEY")

    # prompt_value = template.invoke({"topic": input})

    # print("Stream")
    # for chunk in watsonx_llm.stream(prompt_value):
    #     print(chunk.content, end="")
    # print()
    # print("Stream End")

    # llm_chain = template | watsonx_llm

    db = get_db2_database()
    sql_chain = create_sql_query_chain(watsonx_llm, db)

    # response = sql_chain.invoke({
    #     "question": q
    #     })
    
    # print(response)

    # return db.run(response)

    prompt_template = PromptTemplate.from_file("promptClassify")

    chain = prompt_template | watsonx_llm | JsonOutputParser()
    return chain.invoke({"input": q})