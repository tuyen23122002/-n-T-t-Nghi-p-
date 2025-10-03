# src/flight_booking_agent/services/amadeus_client.py
import os
from amadeus import Client, ResponseError
from dotenv import load_dotenv

load_dotenv()

class AmadeusClient:
    def __init__(self):
        try:
            self.client = Client(
                client_id=os.getenv("AMADEUS_CLIENT_ID"),
                client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
                hostname=os.getenv("AMADEUS_HOSTNAME", "test") # Mặc định là môi trường test
            )
        except Exception as e:
            print(f"Lỗi: Không thể khởi tạo Amadeus Client. Vui lòng kiểm tra biến môi trường.")
            print(f"Chi tiết lỗi: {e}")
            self.client = None

    def search_flights(self, origin, destination, departure_date, adults, non_stop=False, max_results=5):
        if not self.client:
            return {"error": "Amadeus Client chưa được khởi tạo."}

        params = {
            'originLocationCode': origin,
            'destinationLocationCode': destination,
            'departureDate': departure_date,
            'adults': adults,
            'nonStop': 'true' if non_stop else 'false',
            'max': max_results,
            'currencyCode': 'VND'
        }
        
        print(f"--- Đang tìm kiếm chuyến bay với tham số: {params} ---")
        try:
            response = self.client.shopping.flight_offers_search.get(**params)
            return self.format_flight_results(response.data)
        except ResponseError as error:
            print(f"Lỗi API Amadeus: {error.response.result}")
            return {"error": "Không tìm thấy chuyến bay hoặc có lỗi xảy ra.", "details": error.response.result}

    def format_flight_results(self, flight_data):
        """Định dạng lại kết quả cho dễ đọc và xử lý."""
        formatted = []
        for offer in flight_data:
            itinerary = offer['itineraries'][0]
            segment = itinerary['segments'][0]
            price = offer['price']['total']
            
            formatted.append({
                "airline": segment['carrierCode'],
                "flight_number": f"{segment['carrierCode']}{segment['number']}",
                "departure_airport": segment['departure']['iataCode'],
                "departure_time": segment['departure']['at'],
                "arrival_airport": itinerary['segments'][-1]['arrival']['iataCode'],
                "arrival_time": itinerary['segments'][-1]['arrival']['at'],
                "duration": itinerary['duration'],
                "stops": len(itinerary['segments']) - 1,
                "price": float(price),
                "currency": "VND"
            })
        return formatted

# Tạo một instance duy nhất để toàn bộ ứng dụng sử dụng
amadeus_client = AmadeusClient()