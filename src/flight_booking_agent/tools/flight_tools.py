# src/flight_booking_agent/tools/flight_tools.py
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

# Import a-ma-de-us client đã được khởi tạo sẵn
from ..services.amadeus_client import amadeus_client

class FlightSearchInput(BaseModel):
    """Cấu trúc dữ liệu đầu vào cho công cụ tìm kiếm chuyến bay."""
    origin: str = Field(description="Mã IATA của thành phố đi. Ví dụ: 'SGN'")
    destination: str = Field(description="Mã IATA của thành phố đến. Ví dụ: 'HAN'")
    departure_date: str = Field(description="Ngày đi theo định dạng YYYY-MM-DD.")
    adults: Optional[int] = Field(default=1, description="Số lượng hành khách người lớn.")

@tool("flight-search-tool", args_schema=FlightSearchInput)
def search_flights_tool(origin: str, destination: str, departure_date: str, adults: int = 1) -> str:
    """
    Sử dụng công cụ này để tìm kiếm các chuyến bay.
    Công cụ sẽ gọi đến Amadeus API với các tham số được cung cấp.
    """
    print(f"--- TOOL: Bắt đầu tìm kiếm chuyến bay từ {origin} đến {destination} vào ngày {departure_date} cho {adults} người lớn ---")
    
    # Sử dụng amadeus_client đã import
    search_results = amadeus_client.search_flights(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        adults=adults,
        max_results=5 # Giới hạn 5 kết quả cho ngắn gọn
    )
    
    # Tool của LangChain nên trả về một chuỗi (string), vì vậy ta chuyển kết quả thành chuỗi JSON
    if not search_results or "error" in search_results:
        return json.dumps({"error": "Xin lỗi, tôi không tìm thấy chuyến bay nào phù hợp hoặc đã có lỗi xảy ra."})
        
    return json.dumps(search_results, ensure_ascii=False, indent=2)