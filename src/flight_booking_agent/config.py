# src/flight_booking_agent/config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .tools.flight_tools import search_flights_tool

load_dotenv()
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# LLM thông thường để trò chuyện
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base=VLLM_BASE_URL,
    temperature=0.7
)

# LLM đặc biệt có khả năng sử dụng tools
llm_with_tools = llm.bind_tools([search_flights_tool])