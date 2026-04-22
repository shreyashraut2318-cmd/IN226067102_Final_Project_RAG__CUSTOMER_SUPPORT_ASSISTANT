from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import app_graph

app = FastAPI(title="RAG Customer Support Assistant")

# Define the request format 
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Main endpoint that processes the query through the graph workflow.
    """
    try:
        # Source 16: 2-node flow: Input -> Process -> Output
        initial_state = {"query": request.query}
        
        # We execute the graph and get the final state 
        result = app_graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "response": result["response"],
            "category": result["category"] # Support or Escalate 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)