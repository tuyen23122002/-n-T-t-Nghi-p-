# src/flight_booking_agent/agents/manager.py
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from ..graph.state import FlightBookingState

def get_simplified_state(state: FlightBookingState) -> str:
    """Tạo một chuỗi mô tả trạng thái hiện tại một cách đơn giản cho LLM."""
    simplified = {}
    for key, value in state.items():
        if key != "messages" and value is not None:
            simplified[key] = value
    if not simplified:
        return "Chưa có thông tin nào được thu thập."
    return json.dumps(simplified, indent=2, ensure_ascii=False)

def manager_agent_node(state: FlightBookingState, llm: BaseChatModel) -> dict:
    """
    Node này đóng vai trò là Agent Manager, quyết định agent tiếp theo cần gọi.
    """
    print("--- NODE: Agent Manager ---")
    
    # Chuẩn bị context cho prompt
    conversation_history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["messages"]]
    )
    current_state_str = get_simplified_state(state)
    
    system_prompt_template = """
    Bạn là một "Agent Manager" thông minh, điều phối một nhóm các agent chuyên sâu để giúp người dùng đặt vé máy bay.
    Nhiệm vụ của bạn là phân tích lịch sử hội thoại và trạng thái hiện tại, sau đó quyết định agent chuyên sâu nào sẽ thực hiện bước tiếp theo.

    Đây là danh sách các agent chuyên sâu bạn có thể điều phối:
    1. `information_gatherer`: Sử dụng khi thông tin cần thiết (điểm đi, điểm đến, ngày đi) vẫn chưa đầy đủ.
    2. `flight_searcher`: Chỉ sử dụng khi đã có đủ thông tin để tìm kiếm.
    3. `recommendation_agent`: Sử dụng ngay sau khi `flight_searcher` đã tìm thấy kết quả.
    4. `booking_agent`: Sử dụng sau khi người dùng đã xác nhận lựa chọn một chuyến bay cụ thể.
    5. `FINISH`: Sử dụng khi quy trình hoàn tất hoặc không thể tiếp tục.

    QUY TRÌNH SUY NGHĨ CỦA BẠN:
    1. Đọc kỹ tin nhắn cuối cùng của người dùng và xem lại lịch sử hội thoại.
    2. Xem xét trạng thái hiện tại của việc đặt vé.
    3. Dựa trên đó, chọn MỘT agent từ danh sách trên để thực hiện bước tiếp theo.
    4. Chỉ trả lời bằng tên của agent được chọn. Câu trả lời của bạn phải là MỘT trong các chuỗi sau: "information_gatherer", "flight_searcher", "recommendation_agent", "booking_agent", "FINISH".
    """
    
    human_prompt = f"""
    Đây là trạng thái hiện tại:
    {current_state_str}

    Đây là lịch sử hội thoại:
    {conversation_history}

    Dựa trên tất cả thông tin trên, agent tiếp theo nên là gì?
    """
    
    # Gọi LLM
    response = llm.invoke([
        SystemMessage(content=system_prompt_template),
        HumanMessage(content=human_prompt)
    ])
    
    next_agent = response.content.strip()
    print(f">>> MANAGER quyết định: {next_agent}")
    
    # Cập nhật vào state để router có thể sử dụng
    return {"next_agent_to_call": next_agent}