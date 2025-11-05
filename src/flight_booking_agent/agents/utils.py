from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List
from datetime import datetime, timedelta
import re

AIRPORT_MAP = {
    'hà nội': 'HAN', 'hanoi': 'HAN', 'nội bài': 'HAN',
    'hồ chí minh': 'SGN', 'ho chi minh': 'SGN', 'hcm': 'SGN',
    'sài gòn': 'SGN', 'sai gon': 'SGN', 'tân sơn nhất': 'SGN',
    'đà nẵng': 'DAD', 'da nang': 'DAD',
    # Thêm các sân bay khác...
}

def get_iata_code(location_name: str) -> str | None:
    if not location_name: return None
    normalized = location_name.lower().strip()
    
    # Nếu người dùng nhập thẳng mã IATA (3 chữ cái)
    if len(normalized) == 3 and normalized.isalpha():
        return normalized.upper()
        
    return AIRPORT_MAP.get(normalized)

def convert_relative_date(date_str: str) -> str:
    """
    Chuyển đổi các ngày tương đối hoặc định dạng dd/mm sang định dạng 'YYYY-MM-DD'.

    Supported:
        - "hôm nay"
        - "ngày mai"
        - "ngày mốt"
        - "dd/mm" hoặc "dd-mm"
        - "YYYY-mm-dd" (trả về nguyên bản nếu đúng định dạng)
    """
    date_str = date_str.strip().lower()
    today = datetime.now()

    if date_str in ["hôm nay", "hom nay"]:
        return today.strftime("%Y-%m-%d")
    elif date_str in ["ngày mai", "ngay mai"]:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_str in ["ngày mốt", "ngay mot"]:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Kiểm tra định dạng dd/mm hoặc dd-mm
    match = re.match(r"(\d{1,2})[/-](\d{1,2})", date_str)
    if match:
        day, month = map(int, match.groups())
        year = today.year
        try:
            dt = datetime(year, month, day)
            # Nếu ngày đã qua trong năm hiện tại, tự động lấy năm sau
            if dt < today:
                dt = datetime(year + 1, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Ngày không hợp lệ: {date_str}")

    # Kiểm tra xem có phải là ISO format YYYY-MM-DD không
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(f"Không thể nhận dạng ngày: {date_str}")

# Xử lí tin nhắn hội thoại
def filter_for_human_ai(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Lọc lịch sử hội thoại, chỉ giữ lại các tin nhắn từ người dùng (HumanMessage)
    và các tin nhắn trả lời có nội dung của AI (AIMessage).
    
    Loại bỏ:
    - ToolMessage (kết quả từ tool)
    - AIMessage không có nội dung (thường là các tin nhắn chứa tool_calls)
    """
    filtered_messages = []
    for m in messages:
        # Giữ lại tất cả HumanMessage
        if isinstance(m, HumanMessage):
            filtered_messages.append(m)
        # Chỉ giữ lại AIMessage nếu nó có nội dung văn bản để trò chuyện
        elif isinstance(m, AIMessage) and m.content:
            filtered_messages.append(m)
    return filtered_messages