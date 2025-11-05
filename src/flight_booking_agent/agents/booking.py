import json
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional

from ..graph.state import AgentState
from ..config import llm
from ..tools.booking_tools import search_flights_tool
#from ..tools.booking_tools import book_ticket, get_payment_link
#from ..tools.general_tools import get_ancillary
from .utils import get_iata_code, convert_relative_date, filter_for_human_ai

# --- Pydantic model để trích xuất thông tin chuyến bay từ hội thoại ---
class FlightInfoExtractor(BaseModel):
    """Trích xuất thông tin chuyến bay từ tin nhắn của người dùng."""
    departure_city: Optional[str] = Field(None, description="Thành phố hoặc sân bay điểm đi")
    destination_city: Optional[str] = Field(None, description="Thành phố hoặc sân bay điểm đến")
    departure_date: Optional[str] = Field(None, description="Ngày đi, ví dụ: 'hôm nay', 'ngày mai', '25/12'")
    passenger_count: Optional[int] = Field(None, description="Số lượng hành khách người lớn")

# --- LLM và System Prompt ---
booking_llm = llm.bind_tools([
    search_flights_tool,
    #book_ticket,
    #get_payment_link,
    #get_ancillary,
])

SYSTEM_PROMPT = """Bạn là **Vivi**, một trợ lý ảo chuyên hỗ trợ những vấn đề liên quan đến đặt vé máy bay.

        ## Hướng dẫn nghiệp vụ ##
        Nhiệm vụ của bạn là hướng dẫn khách hàng đặt vé theo quy trình CHẶT CHẼ từng bước. TUYỆT ĐỐI KHÔNG BỎ QUA HOẶC NHẢY CÓC CÁC BƯỚC.

        BƯỚC 1.  **Tìm kiếm chuyến bay:**
            * **Đầu tiên:** Thu thập đủ các thông tin sau từ khách hàng:
                * Thành phố khởi hành
                * Thành phố đến
                * Ngày đi
                * Số lượng người lớn (+ trẻ em nếu có)
                * Hạng vé
            * **Khi đã có đủ thông tin trên:** Sử dụng tool `search_flights_tool`.
                * Nếu không tìm thấy chuyến bay: Đề xuất các giải pháp khác (đổi ngày, sân bay, hạng vé). Nếu vẫn không tìm thấy: Báo khách hàng không có chuyến bay phù hợp.
                * Nếu tìm thấy chuyến bay: Hiển thị **TÓM TẮT** danh sách chuyến bay (Giờ bay, Hãng bay, Giá).
            * **Sau khi khách hàng chọn 1 chuyến bay từ danh sách tóm tắt:** Hiển thị **ĐẦY ĐỦ** thông tin chi tiết của chuyến bay đó và **BẮT BUỘC** yêu cầu khách hàng xác nhận lần cuối trước khi chuyển sang bước tiếp theo.

        BƯỚC 2.  **Thông tin hành khách & Dịch vụ bổ sung:**
            * **Sau khi khách hàng xác nhận chuyến bay:** Thu thập thông tin **BẮT BUỘC** cho TẤT CẢ hành khách:
                * Họ và tên đầy đủ
                * Ngày tháng năm sinh
                * Số điện thoại
                *(Thông tin tùy chọn: Giới tính, Email)*
            * **Sau khi đã thu thập đủ thông tin hành khách cho tất cả mọi người:** Hỏi khách hàng có muốn mua thêm dịch vụ bổ sung không.
                * Nếu khách hàng quan tâm: Dùng tool `get_ancillary` để tra cứu và tư vấn các dịch vụ khả dụng kèm giá. 
                * Thu thập lựa chọn dịch vụ của khách hàng.

        BƯỚC 3.  **Thanh toán & Hoàn tất đặt vé:**
            * **NGAY TRƯỚC KHI yêu cầu thanh toán:** **BẮT BUỘC** hiển thị lại **TOÀN BỘ** thông tin đặt vé và yêu cầu khách hàng kiểm tra lại thật kỹ.
            * **Yêu cầu khách hàng phản hồi hợp lệ:** Nếu thông tin đặt vé là đúng, yêu cầu khách hàng phản hồi là `xác nhận` để tiếp tục.
            * **CHỈ KHI** khách hàng `xác nhận` thông tin đúng: sử dụng tool `get_payment_link` và yêu cầu khách hàng phản hồi `đã thanh toán` để nhận mã đặt chỗ.
            * **CHỈ KHI** khác hàng `đã thanh toán` xong qua link trên: Sử dụng tool `book_ticket` để hoàn tất đặt vé và cung cấp mã đặt chỗ cho khách hàng.

        ## Chuyển hướng ##
        Thực hiện chuyển hướng tới `manager_agent`, **TUYỆT ĐỐI** không tự ý trả lời nếu yêu cầu của khách hàng không liên quan đến tìm kiếm chuyến bay, đặt vé.\
        Khi quyết định chuyển hứng, hãy trả về dạng cấu trúc `BookingHandoff`.

        ## Giao tiếp ##
        - Gọi khách là **anh/chị**. Xưng **em** với bản thân.
        - Tránh dùng 'tôi', 'mình' hay gọi khách là 'em'.
        - Văn phong tự nhiên, thân thiện như người thật.
        - Nói chuyện ngắn gọn, rõ ý. Tránh dài dòng, rập khuôn.

        ## QUAN TRỌNG ##
         - Khi nhận được kết quả từ tool (ToolMessage), hãy tóm tắt nó cho người dùng và đề xuất bước tiếp theo.
         - Nếu yêu cầu của khách hàng không liên quan đến đặt vé, hãy nói chính xác câu: "Về vấn đề này, em xin phép chuyển cho một chuyên viên khác." và không làm gì thêm.
        - Luôn giao tiếp thân thiện, gọi khách là anh/chị và xưng em.
"""

# Mọi thứ import và định nghĩa bên trên giữ nguyên

def booking_node(state: AgentState) -> dict:
    """
    Node xử lý toàn bộ nghiệp vụ đặt vé máy bay một cách tuần tự.
    (Phiên bản đã tái cấu trúc theo state và áp dụng pattern update an toàn)
    """
    print("---NODE: BOOKING---")
    # Tạo một bản sao để tránh thay đổi state gốc một cách không mong muốn
    messages = list(state['messages'])
    current_agent = "booking_agent"

    # STATE 1: Xử lý kết quả vừa nhận được từ Tool
    if isinstance(messages[-1], ToolMessage):
        print(">>> Booking Node [State 1]: Đang xử lý kết quả từ Tool...")
        try:
            search_results_data = json.loads(messages[-1].content)
        except json.JSONDecodeError:
            search_results_data = {"error": "Dữ liệu trả về từ tool không hợp lệ."}

        prompt = HumanMessage(
            content=f"""Dựa vào kết quả từ tool call sau đây:
            ---
            {messages[-1].content}
            ---
            Hãy tóm tắt kết quả cho người dùng và hỏi họ muốn chọn chuyến bay nào.
            """
        )
        response = booking_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), prompt])
        
        # <<< SỬA ĐỔI 1 >>>
        updates = {
            "messages": state["messages"] + [response],
            "search_results": search_results_data,
            "previous_agent": current_agent
        }
        return {**state, **updates}

    # STATE 2: Đã có kết quả tìm kiếm, bây giờ xử lý lựa chọn của người dùng
    if state.get("search_results"):
        print(">>> Booking Node [State 2]: Đã có kết quả, xử lý lựa chọn của người dùng...")
        
        filtered_messages = filter_for_human_ai(messages)
        
        contextual_prompt = HumanMessage(
            content=f"""
            Ngữ cảnh: Em vừa gửi cho khách danh sách các chuyến bay sau:
            {json.dumps(state.get("search_results"), ensure_ascii=False, indent=2)}

            Bây giờ, khách hàng trả lời: "{messages[-1].content}"

            Dựa vào câu trả lời của khách, hãy thực hiện bước tiếp theo trong quy trình.
            - Nếu khách CHỌN một chuyến bay: Hãy hiển thị thông tin CHI TIẾT của chuyến đó và YÊU CẦU XÁC NHẬN.
            - Nếu khách hỏi thêm hoặc muốn tìm lại: Hãy trả lời và xử lý yêu cầu đó.
            - Nếu yêu cầu của khách không liên quan: Chuyển hướng cho chuyên viên khác.
            """
        )
        
        final_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + filtered_messages[:-1] + [contextual_prompt]
        response = booking_llm.invoke(final_prompt)

        # <<< SỬA ĐỔI 2 >>>
        updates = {
            "messages": state["messages"] + [response],
            "previous_agent": current_agent
        }
        if "chuyên viên khác" in response.content:
            updates["next_agent"] = "manager"

        return {**state, **updates}

    # STATE 3: Thu thập thông tin và kích hoạt tìm kiếm (chỉ chạy khi chưa có search_results)
    print(">>> Booking Node [State 3]: Thu thập thông tin...")
    structured_llm = llm.with_structured_output(FlightInfoExtractor)
    extractor_prompt = f"Trích xuất thông tin chuyến bay từ câu sau. Input: \"{messages[-1].content}\""
    extracted_info = structured_llm.invoke(extractor_prompt)

    current_info = {
        "departure_from": state.get("departure_from"),
        "arrival_to": state.get("arrival_to"),
        "departure_date": state.get("departure_date"),
        "passenger_count": state.get("passenger_count"),
    }

    if extracted_info.departure_city: current_info["departure_from"] = get_iata_code(extracted_info.departure_city)
    if extracted_info.destination_city: current_info["arrival_to"] = get_iata_code(extracted_info.destination_city)
    if extracted_info.departure_date: current_info["departure_date"] = convert_relative_date(extracted_info.departure_date)
    if extracted_info.passenger_count: current_info["passenger_count"] = extracted_info.passenger_count

    # Nếu đã đủ thông tin, gọi tool
    if all(current_info.values()):
        print(">>> Booking Node: Đủ thông tin, chuẩn bị gọi Tool tìm kiếm...")
        tool_input_message = HumanMessage(
            content=f"Tìm chuyến bay từ {current_info['departure_from']} đến {current_info['arrival_to']} vào ngày {current_info['departure_date']} và cho {current_info['passenger_count']} hành khách."
        )
        response_with_tool_call = booking_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), tool_input_message])
        
        # <<< SỬA ĐỔI 3 >>>
        updates = {
            **current_info,
            "messages": state["messages"] + [response_with_tool_call],
            "previous_agent": current_agent
        }
        return {**state, **updates}
    
    # Nếu chưa đủ thông tin, hỏi người dùng
    required_info_map = {
        "điểm đi": "departure_from", "điểm đến": "arrival_to",
        "ngày đi": "departure_date", "số lượng vé": "passenger_count"
    }
    missing_info = [name for name, key in required_info_map.items() if not current_info.get(key)]

    if missing_info:
        print(f">>> Booking Node: Thiếu thông tin: {missing_info}")
        ask_prompt = f"Dạ, để tìm chuyến bay, anh/chị vui lòng cho em biết thêm về {', '.join(missing_info)} ạ."
        response = AIMessage(content=ask_prompt)

        # <<< SỬA ĐỔI 4 >>>
        updates = {
            **current_info,
            "messages": state["messages"] + [response],
            "previous_agent": current_agent
        }
        return {**state, **updates}

    # Trường hợp dự phòng: Nếu không rơi vào các state trên, trả lời chung chung
    print(">>> Booking Node [Fallback]: Xử lý yêu cầu chung...")
    response = booking_llm.invoke(filter_for_human_ai(messages))

    # <<< SỬA ĐỔI 5 >>>
    updates = {
        "messages": state["messages"] + [response],
        "previous_agent": current_agent
    }
    return {**state, **updates}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # --- GIAI ĐOẠN 3: Xử lý các yêu cầu khác và chuyển hướng ---
    filtered_messages = filter_for_human_ai(messages)
    
    # Sử dụng `filtered_messages` để LLM có ngữ cảnh sạch hơn
    response = booking_llm.invoke(filtered_messages)
    current_agent = "booking_agent"

    if "chuyên viên khác" in response.content:
        return {
            "messages": [response],
            "next_agent": "manager",
            "previous_agent": current_agent
        }

    return {
        "messages": [response],
        "previous_agent": current_agent
    }