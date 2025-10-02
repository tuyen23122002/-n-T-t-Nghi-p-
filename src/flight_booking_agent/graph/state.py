# src/flight_booking_agent/graph/state.py
from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator

# --- Định nghĩa các cấu trúc dữ liệu phụ ---
# Việc này giúp code sạch sẽ và dễ đọc hơn

class FlightInfo(TypedDict):
    """Lưu trữ thông tin chi tiết về một chuyến bay."""
    flight_number: str
    airline: str
    departure_time: str
    arrival_time: str
    price: float
    currency: str

class PassengerInfo(TypedDict):
    """Lưu trữ thông tin của một hành khách."""
    full_name: str
    date_of_birth: str
    passenger_type: str # 'Adult', 'Child', 'Infant'

class BookingInfo(TypedDict):
    """Lưu trữ thông tin của một đơn đặt chỗ."""
    booking_id: str # PNR code
    status: str # 'PENDING_PAYMENT', 'CONFIRMED', 'CANCELLED'
    passengers: List[PassengerInfo]
    flight_details: FlightInfo

# --- Định nghĩa Trạng thái Chính của Đồ thị ---

class FlightBookingState(TypedDict):
    """
    Trạng thái toàn cục của hệ thống Multi-Agent đặt vé máy bay.
    Nó đóng vai trò là "bộ nhớ chung" được chia sẻ và cập nhật bởi tất cả các agent.
    
    Mỗi trường trong State tương ứng với kết quả công việc của một agent chuyên sâu.
    """

    # 1. Quản lý hội thoại chung
    messages: Annotated[List[BaseMessage], operator.add]

    # 2. Kết quả từ Agent Thu thập Thông tin (Information Gatherer)
    departure_city: Optional[str]
    destination_city: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    passenger_count: Optional[int]

    # 3. Kết quả từ Agent Tìm kiếm (Flight Searcher)
    # Danh sách các chuyến bay thô tìm được từ API
    search_results: Optional[List[FlightInfo]]
    
    # 4. Kết quả từ Agent Gợi ý/Phân tích (Recommendation/Filter)
    # Chuyến bay mà người dùng đã lựa chọn
    selected_flight: Optional[FlightInfo]

    # 5. Kết quả từ Agent Quản lý Đặt vé (Booking Manager)
    # Danh sách thông tin hành khách do người dùng cung cấp
    passengers_info: Optional[List[PassengerInfo]]

    # 6. Kết quả từ Agent Quản lý Thanh toán (Payment Manager)
    # Thông tin đơn đặt chỗ sau khi đã giữ chỗ hoặc xác nhận
    booking_info: Optional[BookingInfo]

    # 7. (Tùy chọn) Trường điều hướng cho Agent Manager
    # Giúp Agent Manager quyết định bước tiếp theo một cách tường minh hơn
    # Ví dụ: 'gather_info', 'search', 'book', 'pay'
    next_agent_to_call: Optional[str]