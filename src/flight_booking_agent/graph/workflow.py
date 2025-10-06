# src/flight_booking_agent/graph/workflow.py
from functools import partial
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

# --- 1. Import các thành phần ---
from .state import FlightBookingState
from ..agents.information_gatherer import gather_information_node
from ..agents.flight_searcher import flight_search_node
from ..agents.recommendation import recommendation_node
from ..agents.booking_manager import booking_node
from ..tools.flight_tools import search_flights_tool
from ..tools.booking_tools import create_booking_tool
from ..config import llm, llm_with_tools
from langgraph.prebuilt import ToolNode

# --- 2. Định nghĩa các Node ---
bound_search_node = partial(flight_search_node, llm_with_tools=llm_with_tools)
bound_recommend_node = partial(recommendation_node, llm=llm)
bound_booking_node = partial(booking_node, llm_with_tools=llm_with_tools)
all_tools = [search_flights_tool, create_booking_tool]
tool_node = ToolNode(all_tools)

# --- 3. Bộ định tuyến (Router) quyết định bước tiếp theo ---
def should_continue(state: FlightBookingState) -> str:
    """
    Quyết định hành động tiếp theo dựa trên trạng thái hiện tại.
    """
    print("--- ROUTER: Đang đánh giá trạng thái... ---")
    
    # Ưu tiên 1: Nếu agent vừa yêu cầu gọi tool, hãy thực thi tool
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(">>> ROUTER: Quyết định -> Gọi TOOL")
        # Sender sẽ được dùng để biết tool xong thì quay về node nào
        return "call_tool"

    # Ưu tiên 2: Nếu đã có đủ thông tin, hãy tìm kiếm
    if state.get("departure_city") and state.get("destination_city") and state.get("departure_date"):
        # Chỉ tìm kiếm nếu chưa có kết quả
        if not state.get("search_results"):
            print(">>> ROUTER: Quyết định -> Gọi SEARCH AGENT")
            return "search_agent"
    
    # Mặc định: Nếu không rơi vào các trường hợp trên, nghĩa là thiếu thông tin.
    # Agent thu thập thông tin sẽ chạy và sau đó đồ thị sẽ kết thúc, chờ người dùng nhập liệu.
    print(">>> ROUTER: Quyết định -> DỪNG LẠI (chờ người dùng trả lời)")
    return "end"


# --- 4. Xây dựng Đồ thị (Graph) ---
workflow = StateGraph(FlightBookingState)

# Thêm các node vào đồ thị
workflow.add_node("gather_info_agent", gather_information_node)
workflow.add_node("search_agent", bound_search_node)
workflow.add_node("call_tool", tool_node)

# =========================================================================
# LOGIC MỚI ĐỂ PHÁ VỠ VÒNG LẶP
# =========================================================================

# 1. Bất kỳ input nào cũng bắt đầu bằng việc thu thập thông tin
workflow.set_entry_point("gather_info_agent")

# 2. Sau khi thu thập thông tin, gọi router để quyết định bước tiếp theo
workflow.add_conditional_edges(
    "gather_info_agent",
    should_continue,
    {
        "search_agent": "search_agent",
        "end": END  # Nếu router quyết định 'end', đồ thị sẽ dừng lại
    }
)

# 3. Sau khi tìm kiếm, router lại quyết định
workflow.add_conditional_edges(
    "search_agent",
    should_continue,
    {
        "call_tool": "call_tool",
        "end": END # Sau khi tóm tắt kết quả, cũng dừng lại
    }
)

# 4. Sau khi tool chạy xong, nó phải quay lại node đã gọi nó (search_agent) để tóm tắt
workflow.add_edge("call_tool", "search_agent")

# Biên dịch đồ thị
app = workflow.compile()