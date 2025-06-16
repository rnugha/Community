from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from chatbot.chatbot import Chatbot
import uvicorn

app = FastAPI()
bot = Chatbot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: list | None = None

@app.get("/health")
async def health_check():
    """Simple health check endpoint that returns 'OK'"""
    return PlainTextResponse("OK", status_code=200)

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        chat_history = data.get("chat_history", [])
        
        return await bot.process_message(question, chat_history)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  
        port=8001,       
        reload=True    
    )