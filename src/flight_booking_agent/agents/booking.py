import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional, List

from ..graph.state import AgentState
from ..config import llm
from ..tools.booking_tools import search_flights_tool
from .utils import get_iata_code, convert_relative_date, filter_for_human_ai

class FlightInfoExtractor(BaseModel):
    """Trích xuất thông tin chuyến bay từ tin nhắn của người dùng."""
    departure_city: Optional[str] = Field(None, description="Thành phố hoặc sân bay điểm đi")
    destination_city: Optional[str] = Field(None, description="Thành phố hoặc sân bay điểm đến")
    departure_date: Optional[str] = Field(None, description="Ngày đi, ví dụ: 'hôm nay', 'ngày mai', '25/12'")
    passenger_count: Optional[int] = Field(None, description="Số lượng hành khách người lớn")

# --- Pydantic model để hiểu lựa chọn chuyến bay ---
class FlightChoice(BaseModel):
    """Xác định lựa chọn chuyến bay của người dùng từ một danh sách."""
    choice_index: Optional[int] = Field(None, description="Chỉ số (bắt đầu từ 1) của chuyến bay người dùng chọn.")
    is_confirmed: bool = Field(False, description="True nếu người dùng chắc chắn chọn chuyến bay.")

# --- Pydantic model MỚI để trích xuất thông tin hành khách ---
class PassengerInfo(BaseModel):
    """Thông tin chi tiết của một hành khách."""
    full_name: str = Field(description="Họ và tên đầy đủ của hành khách, viết không dấu. Ví dụ: Nguyen Van A")
    date_of_birth: str = Field(description="Ngày tháng năm sinh theo định dạng DD/MM/YYYY. Ví dụ: 25/12/1990")
    phone_number: str = Field(description="Số điện thoại liên lạc. Ví dụ: 0987654321")

class PassengerInfoExtractor(BaseModel):
    """Trích xuất danh sách thông tin của một hoặc nhiều hành khách."""
    passengers: List[PassengerInfo] = Field(description="Danh sách thông tin các hành khách được cung cấp.")

# --- LLM và System Prompt ---
booking_llm = llm.bind_tools([search_flights_tool])
SYSTEM_PROMPT = """Bạn là **Vivi**, một trợ lý ảo chuyên hỗ trợ những vấn đề liên quan đến đặt vé máy bay.
        Luôn giao tiếp thân thiện, gọi khách là anh/chị và xưng em.
        Luôn tuân thủ chặt chẽ quy trình nghiệp vụ từng bước.
        Khi nhận được kết quả từ tool, hãy tóm tắt nó cho người dùng và đề xuất bước tiếp theo.
        Nếu yêu cầu của khách hàng không liên quan đến đặt vé, hãy nói chính xác câu: "Về vấn đề này, em xin phép chuyển cho một chuyên viên khác." và không làm gì thêm.
"""

def booking_node(state: AgentState) -> dict:
    """
    Node xử lý toàn bộ nghiệp vụ đặt vé máy bay một cách tuần tự.
    (Phiên bản tái cấu trúc với logic if/elif/else để sửa lỗi vòng lặp)
    """
    print("---NODE: BOOKING---")
    messages = list(state['messages'])
    current_agent = "booking_agent"

    # ==============================================================================
    # ƯU TIÊN 1: Xử lý kết quả vừa nhận được từ Tool
    # ==============================================================================
    if isinstance(messages[-1], ToolMessage):
        print(">>> Booking Node [State 1]: Đang xử lý kết quả từ Tool...")
        try:
            search_results_data = json.loads(messages[-1].content)
        except (json.JSONDecodeError, TypeError):
            search_results_data = {"error": "Dữ liệu trả về từ tool không hợp lệ."}

        prompt = HumanMessage(
            content=f"""Dựa vào kết quả từ tool call sau đây:
            ---
            {json.dumps(search_results_data, ensure_ascii=False, indent=2)}
            ---
            Hãy tóm tắt kết quả cho người dùng dưới dạng danh sách gạch đầu dòng, mỗi chuyến bay gồm: Mã hiệu, giờ cất cánh, giờ hạ cánh, và giá vé.
            Đánh số thứ tự cho các chuyến bay bắt đầu từ 1. Sau đó hỏi họ muốn chọn chuyến bay nào.
            """
        )
        response = booking_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), prompt])
        
        # **SỬA LỖI TẠI ĐÂY:** Trả về một dictionary chỉ chứa các trường cần cập nhật
        # LangGraph sẽ tự động merge dict này vào state chung.
        return {
            "messages": [response],
            "search_results": search_results_data, # Quan trọng nhất là bước này
            "previous_agent": current_agent
        }

    # ==============================================================================
    # ƯU TIÊN 2: Đã có kết quả, xử lý lựa chọn chuyến bay của người dùng
    # ==============================================================================
    elif state.get("search_results"):
        print(">>> Booking Node [State 2]: Đã có kết quả, xử lý lựa chọn của người dùng...")
        
        # Dùng LLM có cấu trúc để hiểu lựa chọn của người dùng
        structured_llm_choice = llm.with_structured_output(FlightChoice)
        choice_prompt = f"""Dưới đây là danh sách các chuyến bay đã được cung cấp (đánh số từ 1):
        {json.dumps(state["search_results"], ensure_ascii=False, indent=2)}

        Và đây là phản hồi của người dùng: "{messages[-1].content}"

        Dựa vào phản hồi, hãy xác định xem người dùng đã chọn chuyến bay nào (dựa trên số thứ tự) và đã xác nhận lựa chọn đó chưa.
        """
        user_choice = structured_llm_choice.invoke(choice_prompt)

        # Nếu người dùng đã chọn và xác nhận
        if user_choice.is_confirmed and user_choice.choice_index is not None and 1 <= user_choice.choice_index <= len(state["search_results"]):
            # Trừ 1 vì index của list bắt đầu từ 0
            chosen_flight = state["search_results"][user_choice.choice_index - 1]
            
            passenger_count = state.get("passenger_count", 1)
            instructional_prompt = HumanMessage(
                content=f"""
                [Instruction] Người dùng vừa chọn chuyến bay sau đây:
                {json.dumps(chosen_flight, ensure_ascii=False, indent=2)}

                Nhiệm vụ của bạn bây giờ là:
                1. Dùng văn phong thân thiện, xác nhận lại với người dùng rằng bạn đã ghi nhận lựa chọn của họ (ví dụ: "Dạ em xác nhận anh/chị đã chọn chuyến bay...").
                2. Yêu cầu người dùng cung cấp thông tin cho {passenger_count} hành khách lần lượt bao gồm: Họ và tên, Ngày sinh (theo định dạng DD/MM/YYYY), và Số điện thoại.
                """
            )
            
            # Lọc tin nhắn để giữ ngữ cảnh sạch và thêm chỉ thị vào cuối
            filtered_messages = filter_for_human_ai(messages)[-6:]
            final_prompt =  [SystemMessage(content=SYSTEM_PROMPT)] + filtered_messages + [instructional_prompt]
            
            # Gọi LLM để tạo ra câu trả lời tự nhiên
            response = booking_llm.invoke(final_prompt)
            
            return {
                "messages": [response],
                "confirmed_flight": chosen_flight,
                "search_results": None, # Xóa kết quả tìm kiếm cũ để tránh vào lại state này
                "previous_agent": current_agent
            }
        
        # Nếu người dùng chưa chọn rõ ràng, hỏi lại hoặc xử lý câu hỏi phụ
        else:
            response = booking_llm.invoke(filter_for_human_ai(messages)[-6:])
            return {"messages": [response], "previous_agent": current_agent}
        
         # ==============================================================================
    # STATE 4 (ĐÃ SỬA LỖI TRIỆT ĐỂ): Thu thập thông tin hành khách
    # ==============================================================================
    elif state.get("confirmed_flight") and len(state.get("passengers", [])) < state.get("passenger_count", 1):
        print(f">>> Booking Node [State 4]: Thu thập thông tin hành khách...")
        
        structured_llm_pax = llm.with_structured_output(PassengerInfoExtractor)
        extractor_prompt = f"Trích xuất toàn bộ thông tin hành khách từ nội dung sau. Input: \"{messages[-1].content}\""
        
        try:
            extracted_data = structured_llm_pax.invoke(extractor_prompt)
            newly_extracted_passengers = [p.dict() for p in extracted_data.passengers]
        except Exception:
            newly_extracted_passengers = []

        if not newly_extracted_passengers:
            response = AIMessage(content="Dạ em chưa nhận được thông tin hành khách. Anh/chị vui lòng cung cấp lần lượt Họ tên, Ngày sinh (DD/MM/YYYY), và Số điện thoại ạ.")
            updates = {"messages": state["messages"] + [response], "previous_agent": current_agent}
            return {**state, **updates}

        current_passengers = state.get("passengers", [])
        all_passengers = current_passengers + newly_extracted_passengers
        
        remaining = state.get("passenger_count", 1) - len(all_passengers)

        # KỊCH BẢN 1: Vẫn còn thiếu thông tin -> Tạo tin nhắn và DỪNG LẠI
        if remaining > 0:
            print(f">>> Booking Node [State 4]: Vẫn còn thiếu {remaining} hành khách. Yêu cầu nhập thêm...")
            response_content = f"Dạ em đã ghi nhận thông tin. Anh/chị vui lòng cung cấp thông tin cho {remaining} hành khách còn lại ạ."
            response = AIMessage(content=response_content)
            updates = {
                "messages": state["messages"] + [response],
                "passengers": all_passengers,
                "previous_agent": current_agent
            }
            return {**state, **updates}

        # KỊCH BẢN 2: Đã thu thập đủ thông tin -> KHÔNG tạo tin nhắn và CHẠY TIẾP
        else:
            ### =================================================================
            ### BÊN TRONG ELSE: THỰC HIỆN LOGIC CỦA STATE 5
            ### =================================================================
            print(">>> Booking Node [State 5]: Đã đủ thông tin, bắt đầu tổng kết...")
            
            flight_info = state.get("confirmed_flight", {})
            passenger_info = state.get("passengers", [])
            passenger_count = state.get("passenger_count", 1)
            price_per_ticket = flight_info.get("price", 0)
            total_price = price_per_ticket * passenger_count

            instructional_prompt = HumanMessage(content=f"[INSTRUCTION] Bạn đã thu thập đủ thông tin. Bây giờ là bước cuối cùng trước khi thanh toán.\nDữ liệu đã thu thập:\n- Thông tin chuyến bay (giá này là giá cho 1 người): {json.dumps(flight_info, ensure_ascii=False, indent=2)}\n- Thông tin hành khách: {json.dumps(passenger_info, ensure_ascii=False, indent=2)}\n- Số lượng vé: {passenger_count}\n- **TỔNG CHI PHÍ CUỐI CÙNG (ĐÃ TÍNH TOÁN): {total_price} VND**\n\nNhiệm vụ của bạn:\n1. Hiển thị lại **TOÀN BỘ** thông tin đặt vé trên cho người dùng một cách rõ ràng, mạch lạc, chuyên nghiệp.\n2. **QUAN TRỌNG:** Khi hiển thị phần 'Tổng chi phí', hãy sử dụng con số **TỔNG CHI PHÍ CUỐI CÙNG** đã được tính toán ở trên, không dùng giá vé trong 'Thông tin chuyến bay'.\n3. Yêu cầu người dùng kiểm tra lại thật kỹ các thông tin.\n4. **BẮT BUỘC** phải yêu cầu người dùng phản hồi chính xác bằng từ `xác nhận` để tiếp tục.")
            final_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + filter_for_human_ai(messages)[-6:] + [instructional_prompt]
            response = booking_llm.invoke(final_prompt)
            updates = {"messages": state["messages"] + [response], "final_confirmation_sent": True, "previous_agent": current_agent}
            return {**state, **updates}
  
    # ==============================================================================
    # ƯU TIÊN 3 (CUỐI CÙNG): Thu thập thông tin và kích hoạt tìm kiếm mới
    # ==============================================================================
    else:
        print(">>> Booking Node [State 3]: Thu thập thông tin...")
        structured_llm = llm.with_structured_output(FlightInfoExtractor)
        extractor_prompt = f"Trích xuất thông tin chuyến bay từ câu sau. Input: \"{messages[-1].content}\""
        extracted_info = structured_llm.invoke(extractor_prompt)

        # Tạo một dictionary để cập nhật thông tin
        updates = {}
        if extracted_info.departure_city: updates["departure_from"] = get_iata_code(extracted_info.departure_city)
        if extracted_info.destination_city: updates["arrival_to"] = get_iata_code(extracted_info.destination_city)
        if extracted_info.departure_date: updates["departure_date"] = convert_relative_date(extracted_info.departure_date)
        if extracted_info.passenger_count: updates["passenger_count"] = extracted_info.passenger_count

        # Lấy thông tin hiện tại từ state và cập nhật nó
        current_info = {
            "departure_from": state.get("departure_from"),
            "arrival_to": state.get("arrival_to"),
            "departure_date": state.get("departure_date"),
            "passenger_count": state.get("passenger_count"),
        }
        current_info.update(updates)

        if all(current_info.values()):
            print(">>> Booking Node: Đủ thông tin, chuẩn bị gọi Tool tìm kiếm...")
            tool_input_message = HumanMessage(content=f"Tìm chuyến bay từ {current_info['departure_from']} đến {current_info['arrival_to']} vào ngày {current_info['departure_date']} cho {current_info['passenger_count']} hành khách.")
            response_with_tool_call = booking_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), tool_input_message])
            
            # Cập nhật state với thông tin đã thu thập và tool call
            return {**updates, "messages": [response_with_tool_call], "previous_agent": current_agent}
        
        required_info_map = {"điểm đi": "departure_from", "điểm đến": "arrival_to", "ngày đi": "departure_date", "số lượng vé": "passenger_count"}
        missing_info = [name for name, key in required_info_map.items() if not current_info.get(key)]

        if missing_info:
            print(f">>> Booking Node: Thiếu thông tin: {missing_info}")
            ask_prompt = f"Dạ, để tìm chuyến bay, anh/chị vui lòng cho em biết thêm về {', '.join(missing_info)} ạ."
            response = AIMessage(content=ask_prompt)
            # Cập nhật state với thông tin đã có và tin nhắn hỏi thêm
            return {**updates, "messages": [response], "previous_agent": current_agent}

        # Trường hợp không thể xử lý
        response = AIMessage(content="Dạ em xin lỗi, em chưa hiểu rõ yêu cầu của mình. Anh/chị có thể cho em biết điểm đi, điểm đến và ngày đi được không ạ?")
        return {"messages": [response], "previous_agent": current_agent}