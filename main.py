import os
import dotenv

dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

from langchain_ollama import ChatOllama
local_llm = 'llama3.2:3b'
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format='json')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

from prompts import *

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model='nomic-embed-text-v1.5', inference_mode='local'),
)

retriever = vectorstore.as_retriever(k=3)

import json
from langchain_core.messages import HumanMessage, SystemMessage

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):
    question: str # user quesiton
    generation: str # LLM answer
    web_search: str # binary decision to run web search
    max_retries: int # retry depth max for answer generation
    answers: int # number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]
    
from langchain.schema import Document
from langgraph.graph import END

def retrieve(state):
    print("--RETRIEVING--")
    question = state["question"]
    docs = retriever.invoke(question)
    
    return {"documents": docs}

def generate(state):
    print("--GENERATING--")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    
    rag_prompt_formatted = rag_prompt.format(context=documents, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step+1}

def grade_documents(state):
    print("--GRADING DOCUMENTS--")
    documents = state["documents"]
    question = state["question"]
    
    filtered_docs = []
    web_search_flag="no"
    for document in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=document, question=question)
        result = llm_json_mode.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        
        if grade.lower() == "yes":
            print("--DOCUMENT FOUND RELEVANT--")
            filtered_docs.append(document)
            
        else:
            print("--DOCUMENT FOUND IRRELEVANT--")
            web_search_flag = 'yes'
    
    return {"documents": filtered_docs, "web_search": web_search_flag}
    
def web_search(state):
    print("--WEB SEARCH--")
    
    question = state["question"]
    documents = state["documents"]    
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(document["content"] for document in docs)
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    
    return {"documents": documents}

def route_question(state):
    print("--ROUTING--")
    question = [HumanMessage(state["question"])]
    test_vector_store = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + question)
    source = json.loads(test_vector_store.content)['datasource']
    
    print(f"--ROUTING TO {source}")
    return source

def decide_to_generate(state):
    print("--DECIDING TO GENERATE--")
    
    question = state["question"]
    documents = state["documents"]
    web_search_flag = state["web_search"]
    
    if web_search_flag == 'yes':
        print("--NOT ALL DOCUMENTS RELEVANT: ROUTING TO WEBSEARCH--")
        return "websearch"
    else:
        print("--ALL DOCUMENTS RELEVANT: GENERATING CONTENT")
        return "generate"
    
def grade_generation_and_check_hallucinations(state):
    documents = state["documents"]
    question = state["question"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    loop_step = state["loop_step"]
    
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=documents, generation=generation)
    result = llm_json_mode.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)['binary_score']
    
    if grade.lower() == 'yes':
        print("--GROUNDED IN DOCUMENTS--")
        
        final_answer_grader_prompt_formatted = final_answer_grader_prompt.format(question=question, generation=generation)
        result = llm_json_mode.invoke([SystemMessage(content=final_answer_grader_instructions)] + [HumanMessage(content=final_answer_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        
        if grade.lower() == "yes":
            print("--USEFUL--")
            return "useful"
        elif loop_step <= max_retries:
            print("--NOT USEFUL")
            return "not useful"
        else:
            print("--MAX RETRIES REACHED--")
            return "max retries"
        
    elif loop_step <= max_retries:
        print("--HALLUCINATION FOUND: RETRYING")
        return "not supported"
    else:
        print("--MAX RETRIES REACHED")
        return "max retries"
    
from langgraph.graph import StateGraph

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve"
    }
)

workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate"
    }
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_and_check_hallucinations,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END
    }
)

inputs = {"question": "what is an agent", "max_retries": 3}
graph = workflow.compile()

output = graph.invoke(input=inputs)
print(output['generation'].content)