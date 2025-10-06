# src/flight_booking_agent/agents/recommendation.py
from langchain_core.messages import AIMessage
from ..graph.state import FlightBookingState

def recommendation_node(state: FlightBookingState, llm) -> dict:
    print("--- NODE: Gợi ý chuyến bay (Chưa triển khai) ---")
    # Đây là code giữ chỗ. Logic thực tế sẽ phân tích kết quả tìm kiếm.
    return {"messages": [AIMessage(content="Node gợi ý chuyến bay chưa được triển khai.")]}