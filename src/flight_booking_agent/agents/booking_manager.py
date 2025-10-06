# src/flight_booking_agent/agents/booking_manager.py
from langchain_core.messages import AIMessage
from ..graph.state import FlightBookingState

def booking_node(state: FlightBookingState, llm_with_tools) -> dict:
    print("--- NODE: Quản lý đặt vé (Chưa triển khai) ---")
    # Đây là code giữ chỗ cho logic đặt vé.
    return {"messages": [AIMessage(content="Node quản lý đặt vé chưa được triển khai.")]}