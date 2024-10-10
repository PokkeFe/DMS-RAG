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


# ElasticSearch
from elasticsearch import Elasticsearch, AsyncElasticsearch

# Vector Store / WatsonX connection
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter, FilterOperator, MetadataFilter

from utils import create_sparse_vector_query_with_model, create_sparse_vector_query_with_model_and_filter



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
general_prompt_template = PromptTemplate.from_file("promptGeneral")

wxd_creds = {
    "username": os.environ.get("WXD_USERNAME"),
    "password": os.environ.get("WXD_PASSWORD"),
    "wxdurl": os.environ.get("WXD_URL")
}

async_es_client = AsyncElasticsearch(
    wxd_creds["wxdurl"],
    basic_auth=(wxd_creds["username"], wxd_creds["password"]),
    verify_certs=False,
    request_timeout=3600,
)

# State definitions
class State(TypedDict):
    user_input: str
    graph_output: str

class InputState(TypedDict):
    user_input: str

class SQLGenState(TypedDict):
    sql_query: str

def classify_node(state: State) -> State:
    chain = classify_prompt_template | watsonx_llm | JsonOutputParser()
    response: dict = chain.invoke({"input": state["user_input"]})
    return {"graph_output": response["response_method"]}

def sqlgen_node(state: State) -> SQLGenState:
    db = get_db2_database()
    sql_chain = create_sql_query_chain(watsonx_llm, db)
    response : str = sql_chain.invoke({"question", state["user_input"]})
    return {"sql_query": response}

def sqlexec_node(state: SQLGenState) -> State:
    db = get_db2_database()
    result = db.run_no_throw(state["sql_query"])
    return {"graph_output": str(result)}

def search_knowledge_base(state: State) -> State:
    question         = state["user_input"]
    index_name       = ""
    index_text_field = ""
    es_model_name    = ""
    model_text_field = ""
    num_results      = 1
    es_filters = None

    vector_store = ElasticsearchStore(
        es_client=async_es_client,
        index_name=index_name,
        text_field=index_text_field
    )

    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    if es_filters: 
        filters = MetadataFilters(
                filters=[
                    MetadataFilter(key=k,operator=FilterOperator.EQ, value=v) for k, v in es_filters.items()
            ]
        )
        
        query_engine = index.as_query_engine(
            #text_qa_template=prompt_template,
            similarity_top_k=num_results,
            vector_store_query_mode="sparse",
            vector_store_kwargs={
                "custom_query": create_sparse_vector_query_with_model_and_filter(es_model_name, model_text_field=model_text_field, filters=filters)
            },
        )
    else:
        query_engine = index.as_query_engine(
            #text_qa_template=prompt_template,
            similarity_top_k=num_results,
            vector_store_query_mode="sparse",
            vector_store_kwargs={
                "custom_query": create_sparse_vector_query_with_model(es_model_name, model_text_field=model_text_field)
            },
        )
    # Finally query the engine with the user question
    response = query_engine.query(state["user_input"])
    data_response = {
        "llm_response": response.response,
        "references": [node.to_dict() for node in response.source_nodes]
    }
    return {"graph_output": data_response["llm_response"]}

def general_response_node(state: State) -> State:
    chain = general_prompt_template | watsonx_llm | StrOutputParser()
    return {"graph_output": chain.invoke({"input": state["user_input"]})}

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def classify_node_output_router(state: State) -> Literal["sqlgen_node", "search_knowledge_base", "general_response_node"]:
    match state["user_input"]:
        case "search_knowledge_base":
            return "search_knowledge_base"
        case "query_database":
            return "sqlgen_node"
    return "general_response_node"

def query(user_input: str) -> str:

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

    graph = builder.compile(debug=True)
    print(graph.get_graph().draw_ascii())
    return graph.invoke({"user_input": user_input})