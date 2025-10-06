# src/flight_booking_agent/agents/flight_searcher.py
import json
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from ..graph.state import FlightBookingState

def flight_search_node(state: FlightBookingState, llm_with_tools) -> dict:
    """
    Node này chỉ có MỘT nhiệm vụ: Gọi "flight-search-tool" dựa trên thông tin
    đã có sẵn trong state. Nó không giao tiếp trực tiếp với người dùng.
    """
    print("--- NODE: Tìm kiếm chuyến bay ---")
    messages = state['messages']

    # --- PHẦN 1: Xử lý kết quả từ tool ---
    # Nếu tin nhắn cuối cùng là ToolMessage, nghĩa là tool vừa chạy xong.
    # Nhiệm vụ bây giờ là tóm tắt kết quả đó cho người dùng.
    if isinstance(messages[-1], ToolMessage):
        print(">>> Tóm tắt kết quả từ tool...")
        
        # Tạo một prompt ngắn gọn để LLM tóm tắt
        prompt = f"""
        Dựa vào kết quả tìm kiếm chuyến bay sau đây:
        ---
        {messages[-1].content}
        ---
        Hãy tóm tắt các lựa chọn một cách ngắn gọn, thân thiện cho người dùng.
        Nhấn mạnh vào chuyến bay có giá rẻ nhất nếu có.
        """
        
        # Chỉ cần gọi LLM thường để tạo ra một câu trả lời tự nhiên
        response = llm_with_tools.invoke(prompt) 
        
        # Cập nhật state với kết quả tìm kiếm đã được cấu trúc hóa
        # để các agent sau (như recommendation_agent) có thể sử dụng
        try:
            search_results = json.loads(messages[-1].content)
        except json.JSONDecodeError:
            search_results = {"error": "Dữ liệu trả về từ tool không hợp lệ."}

        return {"messages": [response], "search_results": search_results}

    # --- PHẦN 2: Gọi tool ---
    # Nếu không phải là ToolMessage, nghĩa là chúng ta được gọi lần đầu
    # để thực hiện tìm kiếm.
    print(">>> Chuẩn bị gọi tool tìm kiếm...")
    
    # Lấy thông tin trực tiếp từ state, không cần đọc lại hội thoại
    departure_city = state.get("departure_city")
    destination_city = state.get("destination_city")
    departure_date = state.get("departure_date")

    # Tạo một prompt ngắn gọn, trực tiếp yêu cầu LLM gọi tool
    # Việc này hiệu quả hơn nhiều so với việc đưa cả lịch sử hội thoại
    system_prompt = SystemMessage(content="Bạn là một trợ lý ảo, hãy gọi công cụ tìm kiếm chuyến bay với thông tin được cung cấp.")
    
    # Tạo một HumanMessage giả lập để kích hoạt tool calling
    tool_input_message = HumanMessage(
        content=f"Tìm chuyến bay từ {departure_city} đến {destination_city} vào ngày {departure_date}."
    )

    # Gọi llm_with_tools. Nó sẽ thấy tin nhắn và tự động gọi tool phù hợp.
    response_with_tool_call = llm_with_tools.invoke([system_prompt, tool_input_message])
    
    # Trả về AIMessage chứa yêu cầu gọi tool (tool_calls)
    return {"messages": [response_with_tool_call]}