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

import json


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

classify_prompt_template = PromptTemplate.from_file("promptClassify")

# State definitions
class State(TypedDict):
    user_input: str
    graph_output: str

class InputState(TypedDict):
    user_input: str

def classify_node(state: State) -> State:

    chain = classify_prompt_template | watsonx_llm | JsonOutputParser()
    response: dict = chain.invoke({"input": state["user_input"]})
    return {"graph_output": response["response_method"]}

def sqlgen_node(state: State) -> State:
    return {"graph_output": state["graph_output"]}

def sqlexec_node(state: State) -> State:
    return {"graph_output": state["graph_output"]}

def search_knowledge_base(state: State) -> State:
    return {"graph_output": state["graph_output"]}

def general_response_node(state: State) -> State:
    return {"graph_output": state["graph_output"]}

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def classify_node_output_router(state: State) -> Literal["sqlgen_node", "search_knowledge_base", "general_response_node"]:
    return "search_knowledge_base"

def query(user_input: str) -> str:

    # prompt_value = template.invoke({"topic": input})

    # print("Stream")
    # for chunk in watsonx_llm.stream(prompt_value):
    #     print(chunk.content, end="")
    # print()
    # print("Stream End")

    # llm_chain = template | watsonx_llm

    # db = get_db2_database()
    # sql_chain = create_sql_query_chain(watsonx_llm, db)

    # response = sql_chain.invoke({
    #     "question": q
    #     })
    
    # print(response)

    # return db.run(response)

    builder = StateGraph(State, input=InputState)

    builder.add_node("classify_node", classify_node)
    builder.add_node("search_knowledge_base", search_knowledge_base)
    builder.add_node("sqlgen_node", sqlgen_node)
    builder.add_node("sqlexec_node", sqlexec_node)
    builder.add_node("general_response_node", general_response_node)

    builder.add_edge(START, "classify_node")
    builder.add_conditional_edges("classify_node", classify_node_output_router)

    builder.add_edge("sqlgen_node", "sqlexec_node")
    builder.add_edge("sqlexec_node", END)
    builder.add_edge("search_knowledge_base", END)
    builder.add_edge("general_response_node", END)

    graph = builder.compile()
    print(graph.get_graph().draw_ascii())
    return graph.invoke({"user_input": user_input})