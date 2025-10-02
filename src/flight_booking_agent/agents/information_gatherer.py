# src/flight_booking_agent/agents/information_gatherer.py
from langchain_core.messages import SystemMessage, AIMessage
from flight_booking_agent.config import llm
from flight_booking_agent.graph.state import FlightBookingState

def gather_information_node(state: FlightBookingState) -> dict:
    """
    Node này chịu trách nhiệm thu thập thông tin từ người dùng.
    Nó sẽ hỏi các câu hỏi cho đến khi có đủ thông tin cần thiết.
    
    """
    print("--- NODE: Thu thập thông tin ---")
    
    # Lấy lịch sử tin nhắn từ state
    messages = state['messages']

    # Tạo một system prompt để định hướng cho LLM
    system_prompt = SystemMessage(
        content="""
        Bạn là một trợ lý đặt vé máy bay thân thiện.
        Nhiệm vụ của bạn là thu thập thông tin về ĐIỂM ĐI (departure_city) và ĐIỂM ĐẾN (destination_city).
        
        Dựa vào lịch sử hội thoại, hãy kiểm tra xem thông tin nào còn thiếu.
        - Nếu thiếu cả hai, hãy hỏi cả hai.
        - Nếu chỉ thiếu một, hãy hỏi thông tin còn lại.
        - Nếu đã có đủ cả hai, hãy nói "Cảm ơn bạn, tôi đã có đủ thông tin về chặng bay."
        
        Hãy luôn trả lời một cách ngắn gọn và trực tiếp.
        """
    )
    
    # Thêm system prompt vào đầu cuộc hội thoại để LLM biết vai trò của nó
    conversation = [system_prompt] + messages

    # Gọi LLM
    response = llm.invoke(conversation)

    # Trả về một AIMessage để thêm vào lịch sử
    return {"messages": [response]}