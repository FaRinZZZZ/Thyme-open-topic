from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# โหลด .env ก่อนใช้งาน
load_dotenv()

# สร้าง LLM object
llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

app = FastAPI()

class Message(BaseModel):
    content: str

@app.post("/generate")
async def get_analysis(text: Message):
    try:
        res = llm.invoke(text.content)
        return {"content": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))