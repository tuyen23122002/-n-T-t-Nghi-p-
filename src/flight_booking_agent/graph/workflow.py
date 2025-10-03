# src/flight_booking_agent/graph/workflow.py
from functools import partial
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

# --- 1. Import các thành phần ---

# Import State, các hàm agent, tools, và models
from .state import FlightBookingState
from ..agents.information_gatherer import gather_information_node
from ..agents.flight_searcher import flight_search_node
from ..agents.recommendation import recommendation_node
from ..agents.booking_manager import booking_node
# (Giả định bạn sẽ tạo các file agent này)

from ..tools.flight_tools import search_flights_tool
from ..tools.booking_tools import create_booking_tool 
from ..agents.flight_searcher import flight_search_node # Import node mới
from ..config import llm, llm_with_tools # Import llm_with_tools
from langgraph.prebuilt import ToolNode

from ..config import llm, llm_with_tools
from langgraph.prebuilt import ToolNode

# --- 2. Định nghĩa các Node (Các Agent Chuyên sâu) ---

# Tạo ra các phiên bản "hoàn chỉnh" của mỗi node bằng cách "tiêm" model LLM vào.
# Kỹ thuật này giúp các hàm agent không cần import LLM trực tiếp, tránh lỗi import vòng.
bound_gather_info_node = partial(gather_information_node, llm=llm)
bound_search_node = partial(flight_search_node, llm=llm_with_tools)
bound_recommend_node = partial(recommendation_node, llm=llm)
bound_booking_node = partial(booking_node, llm=llm_with_tools)
# ... thêm các node khác cho payment, etc.

# Tạo một ToolNode duy nhất chứa TẤT CẢ các công cụ mà hệ thống có thể sử dụng.
# Agent Manager sẽ tự động định tuyến đến node này khi cần.
all_tools = [search_flights_tool] # Thêm tool mới vào
tool_node = ToolNode(all_tools)


# --- 3. Định nghĩa "Agent Manager" (Router) ---

# Đây là bộ não của hệ thống. Nó nhìn vào trạng thái hiện tại và quyết định
# agent chuyên sâu nào sẽ được gọi tiếp theo.
def agent_manager_router(state: FlightBookingState) -> str:
    """
    Định tuyến luồng công việc đến agent chuyên sâu phù hợp.
    """
    print("--- AGENT MANAGER: Đang đánh giá trạng thái... ---")
    
    # Ưu tiên 1: Nếu một agent vừa yêu cầu gọi tool, hãy thực thi tool ngay lập tức.
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(">>> MANAGER: Quyết định -> Gọi TOOL")
        return "call_tool"
    
    # Ưu tiên 2: Xử lý quy trình booking & thanh toán
    booking_info = state.get("booking_info")
    if booking_info:
        if booking_info.get("status") == "PENDING_PAYMENT":
             # TODO: Thêm logic cho payment agent
             # print(">>> MANAGER: Quyết định -> Gọi PAYMENT AGENT")
             # return "payment_agent"
             pass
        elif booking_info.get("status") == "CONFIRMED":
            print(">>> MANAGER: Quyết định -> KẾT THÚC (Booking thành công)")
            return "end"

    # Ưu tiên 3: Xử lý khi người dùng đã chọn chuyến bay
    if state.get("selected_flight") and not state.get("booking_info"):
        print(">>> MANAGER: Quyết định -> Gọi BOOKING AGENT")
        return "booking_agent"

    # Ưu tiên 4: Xử lý khi đã có kết quả tìm kiếm nhưng người dùng chưa chọn
    if state.get("search_results") and not state.get("selected_flight"):
        print(">>> MANAGER: Quyết định -> Gọi RECOMMENDATION AGENT")
        return "recommendation_agent"
    
    # Ưu tiên 5: Xử lý khi đã có đủ thông tin để tìm kiếm
    if state.get("departure_city") and state.get("destination_city") and state.get("departure_date"):
        # Chỉ tìm kiếm nếu chưa có kết quả
        if not state.get("search_results"):
            print(">>> MANAGER: Quyết định -> Gọi SEARCH AGENT")
            return "search_agent"
        # Nếu đã có kết quả rồi thì chờ người dùng phản hồi
        else:
             print(">>> MANAGER: Quyết định -> KẾT THÚC (Chờ phản hồi người dùng)")
             return "end"

    # Mặc định: Nếu không rơi vào các trường hợp trên, nghĩa là thiếu thông tin.
    print(">>> MANAGER: Quyết định -> Gọi GATHER INFO AGENT")
    return "gather_info_agent"


# --- 4. Xây dựng Đồ thị (Graph) ---

workflow = StateGraph(FlightBookingState)

# Thêm tất cả các node của các agent chuyên sâu vào đồ thị
workflow.add_node("gather_info_agent", bound_gather_info_node)
workflow.add_node("search_agent", bound_search_node)
workflow.add_node("recommendation_agent", bound_recommend_node)
workflow.add_node("booking_agent", bound_booking_node)
workflow.add_node("call_tool", tool_node)

# Thiết lập điểm bắt đầu của quy trình
# Bất kỳ yêu cầu nào của người dùng cũng sẽ bắt đầu bằng việc Manager đánh giá
# Vì vậy, chúng ta cần một điểm khởi đầu để gọi Manager.
# Cách tốt nhất là bắt đầu từ một agent, ví dụ gather_info_agent.
workflow.set_entry_point("gather_info_agent")

# Thêm các cạnh có điều kiện. Đây là nơi sức mạnh của Manager được thể hiện.
# Sau KHI một agent chạy xong, chúng ta sẽ gọi Manager để quyết định bước TIẾP THEO.
# Chúng ta sẽ tạo một vòng lặp: Agent -> Manager -> Agent khác.
workflow.add_conditional_edges(
    "gather_info_agent", # Bắt đầu từ node này
    agent_manager_router, # Dùng Manager để quyết định
    { # Đây là "bảng chỉ đường"
        "gather_info_agent": "gather_info_agent",
        "search_agent": "search_agent",
        "recommendation_agent": "recommendation_agent",
        "booking_agent": "booking_agent",
        "call_tool": "call_tool",
        "end": END
    }
)
workflow.add_conditional_edges("search_agent", agent_manager_router)
workflow.add_conditional_edges("recommendation_agent", agent_manager_router)
workflow.add_conditional_edges("booking_agent", agent_manager_router)

# Sau khi thực thi tool, luồng công việc phải quay lại Manager để đánh giá kết quả
# và quyết định bước tiếp theo.
workflow.add_conditional_edges(
    "call_tool",
    agent_manager_router,
    # Sau khi tool chạy, không thể quay lại chính nó, nên ta chỉ định các agent khác
    {
        "gather_info_agent": "gather_info_agent",
        "search_agent": "search_agent",
        "recommendation_agent": "recommendation_agent",
        "booking_agent": "booking_agent",
        "end": END
    }
)

# Biên dịch đồ thị thành một ứng dụng có thể chạy được
app = workflow.compile()