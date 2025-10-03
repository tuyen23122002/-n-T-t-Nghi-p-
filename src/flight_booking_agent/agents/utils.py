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