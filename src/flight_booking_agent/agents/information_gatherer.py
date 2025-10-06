# src/flight_booking_agent/agents/information_gatherer.py
import json
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional

from ..config import llm
from ..graph.state import FlightBookingState
from .utils import get_iata_code

# --- Class ExtractedFlightInfo không thay đổi ---
class ExtractedFlightInfo(BaseModel):
    departure_city: Optional[str] = Field(default=None, description="Điểm đi...")
    destination_city: Optional[str] = Field(default=None, description="Điểm đến...")
    departure_date: Optional[str] = Field(default=None, description="Ngày đi...")
    passenger_count: Optional[int] = Field(default=None, description="Số lượng hành khách...")

# --- Hàm convert_relative_date không thay đổi ---
def convert_relative_date(date_str: str) -> str:
    if not date_str: return None
    today = datetime.now()
    if 'ngày mai' in date_str.lower():
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    if 'hôm nay' in date_str.lower():
        return today.strftime('%Y-%m-%d')
    # Sửa đổi nhỏ: Xử lý các định dạng ngày tháng khác nhau
    try:
        # Giả sử người dùng có thể nhập dd/mm
        parsed_date = datetime.strptime(date_str, '%d/%m')
        current_year = today.year
        return parsed_date.replace(year=current_year).strftime('%Y-%m-%d')
    except ValueError:
        return date_str


def gather_information_node(state: FlightBookingState) -> dict:
    messages = state.get('messages', [])

    # Nếu chưa có nhiều tin nhắn -> chào hỏi
    if len(messages) <= 1:
        welcome_message = AIMessage(content="Xin chào! Tôi có thể giúp gì cho bạn trong việc tìm kiếm chuyến bay ạ?")
        print(f">>> Lời chào đầu tiên: {welcome_message.content}")
        return {"messages": [welcome_message]}

    # chuẩn bị structured LLM
    structured_llm = llm.with_structured_output(ExtractedFlightInfo)

    extraction_prompt = SystemMessage(
        content="""
        Bạn là một robot trích xuất dữ liệu cực kỳ chính xác. Nhiệm vụ của bạn là phân tích tin nhắn CUỐI CÙNG của người dùng và điền vào các trường thông tin.

        **CÁC QUY TẮC VÀNG:**
        1.  **QUY TẮC ƯU TIÊN KHI NHẬP NHẰNG:** Nếu người dùng chỉ cung cấp TÊN MỘT THÀNH PHỐ (ví dụ: "hà nội", "đà nẵng") mà không có từ khóa "từ" hoặc "đến", bạn **BẮT BUỘC** phải coi đó là **ĐIỂM ĐI (departure_city)** và để **ĐIỂM ĐẾN (destination_city) là `null`**. TUYỆT ĐỐI KHÔNG được điền cùng một thành phố vào cả hai trường.
        2.  **KHÔNG SUY DIỄN:** Nếu người dùng chỉ nói "từ hà nội", bạn phải để `destination_city` là `null`.
        3.  **KHÔNG TỰ ĐỘNG ĐIỀN NGÀY:** Nếu người dùng không đề cập ngày, `departure_date` phải là `null`. Ngày hiện tại ({current_date}) chỉ dùng để tham khảo.
        4.  **CHỈ XỬ LÝ TIN NHẮN CUỐI:** Hoàn toàn bỏ qua các tin nhắn trước đó.

        **VÍ DỤ:**
        -   Tin nhắn người dùng: "hà nội" -> Kết quả: {{ "departure_city": "hà nội", "destination_city": null }}
        -   Tin nhắn người dùng: "tôi muốn đến đà nẵng" -> Kết quả: {{ "destination_city": "đà nẵng", "departure_city": null }}
        -   Tin nhắn người dùng: "2 người" -> Kết quả: {{ "passenger_count": 2 }}
        """.format(current_date=datetime.now().strftime('%Y-%m-%d'))
    )

    # --- Lấy tin nhắn cuối cùng của NGƯỜI DÙNG (very important)
    last_human = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m
            break
        # nếu messages lưu dicts hoặc objects khác, thử kiểm tra attribute 'role' hoặc 'type'
        if hasattr(m, "role") and getattr(m, "role") in ("user", "human"):
            last_human = HumanMessage(content=getattr(m, "content", str(m)))
            break
    if last_human is None:
        # fallback: dùng phần tử cuối cùng (an toàn hơn so với dùng assistant message)
        last_item = messages[-1]
        last_human = HumanMessage(content=getattr(last_item, "content", str(last_item)))

    # Gọi LLM chỉ với tin nhắn người dùng (không truyền nguyên conversation để tránh ghi đè)
    extracted = structured_llm.invoke([extraction_prompt, last_human])

    # ensure we have a ExtractedFlightInfo instance
    if isinstance(extracted, dict):
        extracted_data = ExtractedFlightInfo.parse_obj(extracted)
    elif isinstance(extracted, ExtractedFlightInfo):
        extracted_data = extracted
    else:
        # fallback: nếu là BaseModel của pydantic
        try:
            extracted_data = ExtractedFlightInfo.parse_obj(extracted)
        except Exception:
            extracted_data = ExtractedFlightInfo()  # trống, ko crash

    print(f">>> Thông tin trích xuất được: {extracted_data.dict()}")

    # -------------------------------------------------------------------------
    # Merge với state hiện có (không ghi đè trừ khi có giá trị mới)
    # -------------------------------------------------------------------------
    current_known_info = {
        "departure_city": state.get('departure_city'),
        "destination_city": state.get('destination_city'),
        "departure_date": state.get('departure_date'),
        "passenger_count": state.get('passenger_count'),
    }

    # Nếu LLM trả city, thử convert sang IATA trước, nếu không có IATA thì giữ tên
    if extracted_data.departure_city:
        iata = get_iata_code(extracted_data.departure_city)
        current_known_info['departure_city'] = iata or extracted_data.departure_city

    if extracted_data.destination_city:
        iata = get_iata_code(extracted_data.destination_city)
        current_known_info['destination_city'] = iata or extracted_data.destination_city

    if extracted_data.departure_date:
        current_known_info['departure_date'] = convert_relative_date(extracted_data.departure_date)

    if extracted_data.passenger_count is not None:
        # cast nếu cần
        try:
            current_known_info['passenger_count'] = int(extracted_data.passenger_count)
        except Exception:
            current_known_info['passenger_count'] = extracted_data.passenger_count

    # Kiểm tra đã đủ thông tin chưa
    if all([current_known_info.get('departure_city'),
            current_known_info.get('destination_city'),
            current_known_info.get('departure_date'),
            current_known_info.get('passenger_count')]):
        print(">>> Đã đủ thông tin. Chuyển sang bước tiếp theo.")
        # trả về full info (không cần append message nữa vì không cần hỏi)
        return current_known_info

    # Tạo câu hỏi ngắn gọn nhắm vào thông tin còn thiếu
    missing_info = []
    if not current_known_info.get('departure_city'): missing_info.append("điểm đi")
    if not current_known_info.get('destination_city'): missing_info.append("điểm đến")
    if not current_known_info.get('departure_date'): missing_info.append("ngày đi")
    if not current_known_info.get('passenger_count'): missing_info.append("số lượng hành khách")

    known_info_str = "\n".join([f"- Đã biết {k.replace('_',' ')}: {v}" for k,v in current_known_info.items() if v])

    generation_prompt = f"""
    Bạn là một trợ lý AI hiệu quả. Nhiệm vụ **BẮT BUỘC** của bạn là tạo ra MỘT câu hỏi duy nhất, ngắn gọn để thu thập chính xác những thông tin còn thiếu.

    THÔNG TIN ĐÃ BIẾT:
    {known_info_str if known_info_str else "Chưa có thông tin nào."}

    THÔNG TIN CÒN THIẾU:
    {', '.join(missing_info)}

    HÃY TRẢ LỜI MỘT CÂU HỎI NGẮN GỌN, KHÔNG CHÀO HỎI.
    """

    ai_response = llm.invoke([
        SystemMessage(content="Bạn là một trợ lý AI, chỉ nói những gì được yêu cầu trong prompt."),
        HumanMessage(content=generation_prompt)
    ])

    print(f">>> Câu hỏi được tạo ra: {ai_response.content}")

    # Append assistant message (không ghi đè messages cũ)
    new_messages = list(messages)  # copy
    new_messages.append(AIMessage(content=ai_response.content))

    # Trả về cập nhật: các trường hiện có + messages mới (framework sẽ merge vào state)
    out = {
        "departure_city": current_known_info.get('departure_city'),
        "destination_city": current_known_info.get('destination_city'),
        "departure_date": current_known_info.get('departure_date'),
        "passenger_count": current_known_info.get('passenger_count'),
        "messages": new_messages,
    }

    return out
