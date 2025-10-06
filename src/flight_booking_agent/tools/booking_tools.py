# src/flight_booking_agent/tools/booking_tools.py
from langchain_core.tools import tool

@tool("create-booking-tool")
def create_booking_tool(flight_id: str, passenger_details: list) -> str:
    """
    _PLACEHOLDER_ Giả lập việc tạo một đơn đặt vé.
    Trong ứng dụng thực tế, hàm này sẽ gọi đến API đặt vé.
    """
    print("--- TOOL: Đang tạo booking (Chưa triển khai) ---")
    return '{"booking_id": "DUMMY_PNR_12345", "status": "PENDING_PAYMENT"}'