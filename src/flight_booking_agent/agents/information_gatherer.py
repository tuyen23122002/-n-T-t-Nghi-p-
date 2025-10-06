# src/flight_booking_agent/agents/information_gatherer.py
import json
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional

from ..config import llm
from ..graph.state import FlightBookingState
from .utils import get_iata_code

# --- Cấu trúc trích xuất thông tin (Giữ nguyên) ---
class ExtractedFlightInfo(BaseModel):
    departure_city: Optional[str] = Field(description="Điểm đi. Trả về None nếu không được đề cập.")
    destination_city: Optional[str] = Field(description="Điểm đến. Trả về None nếu không được đề cập.")
    departure_date: Optional[str] = Field(description="Ngày đi. Chuyển 'ngày mai' thành YYYY-MM-DD. Trả về None nếu không được đề cập.")
    passenger_count: Optional[int] = Field(description="Số lượng hành khách. Trả về None nếu không được đề cập.")

def convert_relative_date(date_str: str) -> str:
    if not date_str: return None
    today = datetime.now()
    if 'ngày mai' in date_str.lower():
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    if 'hôm nay' in date_str.lower():
        return today.strftime('%Y-%m-%d')
    return date_str

def gather_information_node(state: FlightBookingState) -> dict:
    print("--- NODE: Thu thập thông tin (Chống Nhập Nhằng) ---")
    
    messages = state['messages']
    structured_llm = llm.with_structured_output(ExtractedFlightInfo)

    # ==============================================================================
    # SỬA ĐỔI: PROMPT CUỐI CÙNG, THÊM QUY TẮC XỬ LÝ NHẬP NHẰNG
    # ==============================================================================
    extraction_prompt = SystemMessage(
        content="""
        Bạn là một robot trích xuất dữ liệu cực kỳ chính xác. Nhiệm vụ của bạn là phân tích tin nhắn CUỐI CÙNG của người dùng và điền vào các trường thông tin.

        **CÁC QUY TẮC VÀNG:**
        1.  **QUY TẮC ƯU TIÊN KHI NHẬP NHẰNG:** Nếu người dùng chỉ cung cấp TÊN MỘT THÀNH PHỐ (ví dụ: "hà nội", "đà nẵng") mà không có từ khóa "từ" hoặc "đến", bạn **BẮT BUỘC** phải coi đó là **ĐIỂM ĐI (departure_city)** và để **ĐIỂM ĐẾN (destination_city) là `null`**. TUYỆT ĐỐI KHÔNG được điền cùng một thành phố vào cả hai trường.
        2.  **KHÔNG SUY DIỄN:** Nếu người dùng chỉ nói "từ hà nội", bạn phải để `destination_city` là `null`.
        3.  **KHÔNG TỰ ĐỘNG ĐIỀN NGÀY:** Nếu người dùng không đề cập ngày, `departure_date` phải là `null`. Ngày hiện tại ({current_date}) chỉ dùng để tham khảo.
        4.  **CHỈ XỬ LÝ TIN NHẮN CUỐI:** Hoàn toàn bỏ qua các tin nhắn trước đó.

        **VÍ DỤ:**
        -   Tin nhắn người dùng: "hà nội" -> Kết quả: `{{ "departure_city": "hà nội", "destination_city": null }}`
        -   Tin nhắn người dùng: "tôi muốn đến đà nẵng" -> Kết quả: `{{ "destination_city": "đà nẵng", "departure_city": null }}`
        -   Tin nhắn người dùng: "2 người" -> Kết quả: `{{ "passenger_count": 2 }}`
        """.format(current_date=datetime.now().strftime('%Y-%m-%d'))
    )
    
    extracted_data: ExtractedFlightInfo = structured_llm.invoke([extraction_prompt, messages[-1]])
    print(f">>> Thông tin trích xuất được: {extracted_data.dict()}")

    # --- Phần còn lại của code giữ nguyên logic cũ, vì nó đã đúng ---

    updates = {}
    if extracted_data.departure_city and not state.get('departure_city'):
        updates['departure_city'] = get_iata_code(extracted_data.departure_city)
    if extracted_data.destination_city and not state.get('destination_city'):
        updates['destination_city'] = get_iata_code(extracted_data.destination_city)
    if extracted_data.departure_date and not state.get('departure_date'):
        updates['departure_date'] = convert_relative_date(extracted_data.departure_date)
    if extracted_data.passenger_count and not state.get('passenger_count'):
        updates['passenger_count'] = extracted_data.passenger_count
    
    temp_state = {**state, **updates}
    
    if (temp_state.get('departure_city') and 
        temp_state.get('destination_city') and 
        temp_state.get('departure_date') and
        temp_state.get('passenger_count')):
        print(">>> Đã đủ thông tin. Chuyển sang bước tiếp theo.")
        return updates

    missing_info = []
    if not temp_state.get('departure_city'): missing_info.append("điểm đi")
    if not temp_state.get('destination_city'): missing_info.append("điểm đến")
    if not temp_state.get('departure_date'): missing_info.append("ngày đi")
    if not temp_state.get('passenger_count'): missing_info.append("số lượng hành khách")

    current_info_str = json.dumps({
        "Điểm đi đã biết": temp_state.get('departure_city'),
        "Điểm đến đã biết": temp_state.get('destination_city'),
        "Ngày đi đã biết": temp_state.get('departure_date'),
        "Số lượng hành khách đã biết": temp_state.get('passenger_count'),
    }, indent=2, ensure_ascii=False)

    generation_prompt = f"""
    Bạn là một trợ lý đặt vé máy bay thân thiện.
    Nhiệm vụ của bạn là hỏi người dùng MỘT câu hỏi duy nhất để thu thập thông tin còn thiếu.
    Thông tin đã có:
    {current_info_str}
    Thông tin còn thiếu: {", ".join(missing_info)}.
    Dựa vào đó, hãy tạo ra MỘT câu hỏi tự nhiên và ngắn gọn để lấy thông tin tiếp theo.
    Câu hỏi của bạn:
    """
    
    ai_response = llm.invoke([
        SystemMessage(content="Bạn là một trợ lý đặt vé máy bay thân thiện."),
        HumanMessage(content=generation_prompt)
    ])
    
    print(f">>> Câu hỏi được tạo ra: {ai_response.content}")
    
    updates['messages'] = [AIMessage(content=ai_response.content)]
    
    return updates