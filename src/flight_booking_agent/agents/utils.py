from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List
from datetime import datetime, timedelta
import re

AIRPORT_MAP = {
    # Việt Nam
    'hà nội': 'HAN', 'hanoi': 'HAN', 'nội bài': 'HAN',
    'hồ chí minh': 'SGN', 'ho chi minh': 'SGN', 'hcm': 'SGN',
    'sài gòn': 'SGN', 'sai gon': 'SGN', 'tân sơn nhất': 'SGN',
    'đà nẵng': 'DAD', 'da nang': 'DAD',
    'hải phòng': 'HPH', 'cat bi': 'HPH',
    'huế': 'HUI', 'phú bài': 'HUI',
    'nha trang': 'CXR', 'cam ranh': 'CXR',
    'phú quốc': 'PQC', 'phu quoc': 'PQC',
    'cần thơ': 'VCA', 'can tho': 'VCA',

    # Đông Nam Á
    'bangkok': 'BKK', 'suvarnabhumi': 'BKK',
    'don muang': 'DMK',
    'singapore': 'SIN', 'changi': 'SIN',
    'kuala lumpur': 'KUL',
    'jakarta': 'CGK',
    'manila': 'MNL',
    'phnom penh': 'PNH',
    'siem reap': 'REP',

    # Đông Á
    'tokyo': 'NRT', 'narita': 'NRT',
    'haneda': 'HND',
    'osaka': 'KIX',
    'seoul': 'ICN', 'incheon': 'ICN',
    'busan': 'PUS',
    'đài bắc': 'TPE', 'taipei': 'TPE',
    'hong kong': 'HKG',
    'thượng hải': 'PVG', 'shanghai': 'PVG',
    'bắc kinh': 'PEK', 'beijing': 'PEK',

    # Trung Đông
    'dubai': 'DXB',
    'abu dhabi': 'AUH',
    'doha': 'DOH',
    'istanbul': 'IST',

    # Châu Âu
    'london': 'LHR', 'heathrow': 'LHR',
    'paris': 'CDG', 'charles de gaulle': 'CDG',
    'frankfurt': 'FRA',
    'munich': 'MUC',
    'madrid': 'MAD',
    'barcelona': 'BCN',
    'rome': 'FCO', 'fiumicino': 'FCO',
    'amsterdam': 'AMS',
    'zurich': 'ZRH',
    'vienna': 'VIE',

    # Bắc Mỹ
    'new york': 'JFK', 'jfk': 'JFK',
    'los angeles': 'LAX', 'la': 'LAX',
    'san francisco': 'SFO',
    'chicago': 'ORD',
    'toronto': 'YYZ',
    'vancouver': 'YVR',

    # Úc
    'sydney': 'SYD',
    'melbourne': 'MEL',
    'brisbane': 'BNE'
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