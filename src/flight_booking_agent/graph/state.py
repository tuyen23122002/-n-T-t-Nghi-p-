from typing import List, TypedDict, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    Trạng thái chung cho hệ thống Multi-Agent, được chia sẻ qua tất cả các node.

    Attributes:
        messages: Lịch sử hội thoại.
        next_agent: Agent tiếp theo được gọi, do Manager quyết định.
        ... các trường dữ liệu khác được thu thập trong quá trình ...
    """
    # Lịch sử hội thoại
    messages: Annotated[List[BaseMessage], operator.add]

    # Trường điều hướng luồng
    next_agent: Optional[str]
    # Lưu lại tên của agent vừa hoạt động ở lượt trước.
    previous_agent: Optional[str]

    # Các thông tin được thu thập (tương đương BookingDeps)
    airline: Optional[str]
    flight_number: Optional[str]
    departure_from: Optional[str]
    arrival_to: Optional[str]
    departure_date: Optional[str]
    departure_time: Optional[str]
    travel_class: Optional[Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]]
    passenger_count: Optional[List[dict]]
    ancillary: Optional[List[dict]]
    original_ticket_price: Optional[str]
    
    # Kết quả tìm kiếm và lựa chọn
    search_results: Optional[List[dict]]
    confirmed_flight: Optional[dict]

    # THÊM TRƯỜNG MỚI ĐỂ LƯU DANH SÁCH HÀNH KHÁCH
    passengers: Optional[List[dict]]
    
     # THÊM TRƯỜNG MỚI
    final_confirmation_sent: Optional[bool] # Cờ để đánh dấu đã gửi xác nhận cuối cùng