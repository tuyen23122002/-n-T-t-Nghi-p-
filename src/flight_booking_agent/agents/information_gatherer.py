# src/flight_booking_agent/agents/information_gatherer.py
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

from flight_booking_agent.config import llm
from flight_booking_agent.graph.state import FlightBookingState
from .utils import get_iata_code # Đảm bảo bạn đã tạo file utils.py

# --- Bước 1: Định nghĩa "biểu mẫu" để LLM điền vào ---
# Đây là tất cả các thông tin chúng ta muốn trích xuất từ người dùng.
class ExtractedFlightInfo(BaseModel):
    """Trích xuất thông tin chi tiết về chuyến bay từ cuộc hội thoại của người dùng."""
    departure_city: Optional[str] = Field(
        description="Điểm đi, ví dụ: 'Hà Nội' hoặc 'SGN'. Bỏ trống nếu không được đề cập."
    )
    destination_city: Optional[str] = Field(
        description="Điểm đến, ví dụ: 'Sài Gòn' hoặc 'SGN'. Bỏ trống nếu không được đề cập."
    )
    departure_date: Optional[str] = Field(
        description="Ngày đi. Cần chuyển đổi các từ như 'ngày mai', 'hôm nay' thành định dạng YYYY-MM-DD."
    )
    return_date: Optional[str] = Field(
        description="Ngày về cho chuyến bay khứ hồi (nếu có). Cần chuyển đổi thành định dạng YYYY-MM-DD."
    )
    passenger_count: Optional[int] = Field(
        description="Tổng số lượng hành khách. Chỉ lấy số nguyên từ câu nói, ví dụ '2 người' -> 2."
    )
    # Bạn có thể thêm các trường tùy chọn khác ở đây, ví dụ: travel_class

def convert_relative_date(date_str: str) -> str:
    """Hàm phụ trợ để chuyển đổi 'ngày mai', 'hôm nay' thành YYYY-MM-DD."""
    if not date_str: return None
    today = datetime.now()
    if 'ngày mai' in date_str.lower():
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    if 'hôm nay' in date_str.lower():
        return today.strftime('%Y-%m-%d')
    # Giả định người dùng đã nhập đúng định dạng nếu không phải từ tương đối
    # Một hệ thống thực tế sẽ cần trình phân tích ngày tháng mạnh mẽ hơn
    return date_str

def gather_information_node(state: FlightBookingState) -> dict:
    """
    Trích xuất thông tin từ tin nhắn người dùng, cập nhật vào state,
    và hỏi các câu hỏi tiếp theo cho đến khi đủ thông tin bắt buộc.
    """
    print("--- NODE: Thu thập và Trích xuất thông tin ---")
    
    messages = state['messages']
    
    # Tạo một LLM có khả năng trích xuất dữ liệu theo cấu trúc của ExtractedFlightInfo
    structured_llm = llm.with_structured_output(ExtractedFlightInfo)

    # System prompt mới, tập trung vào việc trích xuất
    system_prompt = SystemMessage(
        content="""
        Bạn là một chuyên gia phân tích hội thoại, nhiệm vụ của bạn là trích xuất thông tin
        về chuyến bay từ tin nhắn của người dùng. Hãy điền vào các trường thông tin
        bạn tìm thấy. Nếu không tìm thấy, hãy để trống.
        Hôm nay là ngày {current_date}.
        """.format(current_date=datetime.now().strftime('%Y-%m-%d'))
    )
    
    # Chỉ cần tin nhắn cuối cùng của người dùng để trích xuất thông tin mới
    user_last_message = messages[-1]
    
    # Gọi LLM để trích xuất
    extracted_data: ExtractedFlightInfo = structured_llm.invoke([system_prompt, user_last_message])
    print(f">>> Thông tin trích xuất được: {extracted_data.dict()}")

    # --- Bước 2: Cập nhật thông tin vào State ---
    # Chỉ cập nhật nếu state chưa có giá trị và LLM trích xuất được giá trị mới
    
    # Tạo một dictionary để chứa các cập nhật
    updates = {}
    
    if extracted_data.departure_city and not state.get('departure_city'):
        updates['departure_city'] = get_iata_code(extracted_data.departure_city)
    
    if extracted_data.destination_city and not state.get('destination_city'):
        updates['destination_city'] = get_iata_code(extracted_data.destination_city)
    
    if extracted_data.departure_date and not state.get('departure_date'):
        updates['departure_date'] = convert_relative_date(extracted_data.departure_date)

    if extracted_data.passenger_count and not state.get('passenger_count'):
        updates['passenger_count'] = extracted_data.passenger_count
    
    # Bạn có thể thêm các trường tùy chọn như return_date ở đây
    if extracted_data.return_date and not state.get('return_date'):
        updates['return_date'] = convert_relative_date(extracted_data.return_date)

    # --- Bước 3: Kiểm tra và hỏi thông tin còn thiếu ---
    # Sau khi cập nhật, kiểm tra lại state để quyết định câu hỏi tiếp theo
    
    # Tạo một bản state tạm thời đã được cập nhật để kiểm tra
    temp_state = {**state, **updates}
    
    ai_response_content = None
    if not temp_state.get('departure_city'):
        ai_response_content = "Chào bạn! Để bắt đầu, bạn muốn bay từ đâu ạ?"
    elif not temp_state.get('destination_city'):
        ai_response_content = f"OK, điểm đi là {temp_state['departure_city']}. Bạn muốn bay đến thành phố nào?"
    elif not temp_state.get('departure_date'):
        ai_response_content = f"Đã ghi nhận chặng bay từ {temp_state['departure_city']} đến {temp_state['destination_city']}. Bạn muốn khởi hành vào ngày nào?"
    elif not temp_state.get('passenger_count'):
        ai_response_content = "Chuyến đi này có bao nhiêu hành khách ạ?"

    # Nếu có câu hỏi cần hỏi, thêm nó vào danh sách tin nhắn
    if ai_response_content:
        updates['messages'] = [AIMessage(content=ai_response_content)]
    
    # Trả về tất cả các cập nhật cho state
    # LangGraph sẽ tự động hợp nhất dictionary này vào state hiện tại
    return updates