from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from src.ingestion import get_retriever
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv # New import

# Load the .env file
load_dotenv()

# Get the key from the environment instead of hard-coding
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pass it to the LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

class AgentState(TypedDict):
    query: str
    context: str
    response: str
    category: str 

def retrieve_node(state: AgentState):
    # Source 40: Retriever implementation
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    return {"context": "\n".join([doc.page_content for doc in docs])}

def processing_node(state: AgentState):
    """
    Refined Node: Forces the LLM to follow RAG constraints.
    """
    # Source 77: Escalation criteria - No context found by retriever
    if not state["context"].strip():
        return {"category": "escalate", "response": "I cannot find this in our database."}

    # Source 103: Prompt design logic - Strictly limiting the LLM
    prompt = f"""
    SYSTEM ROLE: You are a Customer Support Bot for a specific company. 
    RULE 1: ONLY use the 'Context' provided below. 
    RULE 2: If the answer is NOT in the context, you MUST reply with the exact word: ESCALATE.
    RULE 3: Do NOT use your own outside knowledge (like recipes or general facts).

    Context: {state['context']}
    User Query: {state['query']}
    
    Response:"""
    
    answer = llm.invoke(prompt).content
    
    # Source 106: Routing decision based on intent/confidence
    if "ESCALATE" in answer.upper():
        return {"category": "escalate", "response": "Passing to human expert..."}
    
    return {"category": "support", "response": answer}

def human_node(state: AgentState):
    # Source 19 & 81: HITL Design
    print("\n--- HITL Escalation Triggered ---")
    human_ans = input("Human Agent, provide answer: ")
    return {"response": f"Human Expert: {human_ans}"}

def router(state: AgentState):
    # Source 43: Routing Layer
    return "human" if state["category"] == "escalate" else "end"

# Source 15: Graph-based workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("process", processing_node)
workflow.add_node("human", human_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "process")
workflow.add_conditional_edges("process", router, {"human": "human", "end": END})
workflow.add_edge("human", END)

app_graph = workflow.compile()