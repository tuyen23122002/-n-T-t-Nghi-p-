# src/flight_booking_agent/config.py
import os
from dotenv import load_dotenv
# Thay đổi import
from langchain_google_vertexai import ChatVertexAI



load_dotenv()
llm = ChatVertexAI(
    model_name="gemini-2.5-pro", 
    temperature=0.7,
    project="end-to-end-agentic-rag",  
    location="us-central1",     
)


