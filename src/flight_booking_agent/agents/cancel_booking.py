# src/flight_booking_agent/agents/cancel_booking.py
from langchain_core.messages import AIMessage
from ..graph.state import AgentState

# Giả lập các tool có thể cần
# from ..tools.booking_tools import cancel_booking

# Giả lập llm
# from ..config import llm

def cancel_booking_node(state: AgentState) -> dict:
    """
    NODE GIẢ LẬP: Xử lý nghiệp vụ hủy vé.
    Trong tương lai, node này sẽ chứa logic để gọi tool hủy vé.
    """
    print("---NODE: CANCEL BOOKING (Placeholder)---")
    
    # Hiện tại, chỉ trả về một tin nhắn thông báo
    response_message = AIMessage(content="Dạ, để hủy vé, anh/chị vui lòng cung cấp mã đặt chỗ ạ.")
    
    return {
        "messages": [response_message],
        "previous_agent": "cancel_booking_agent"
    }