from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
#from langgraph.checkpoint.sqlite import SqliteSaver  # THÊM DÒNG NÀY
from langgraph.checkpoint.memory import InMemorySaver
from .state import AgentState
from ..agents import manager, booking, cancel_booking, general, router
from ..tools import booking_tools

# 1. Tập hợp tất cả các tool
all_tools = [
    booking_tools.search_flights_tool,
]
tool_node = ToolNode(all_tools)

# 2. Xây dựng đồ thị
workflow = StateGraph(AgentState)

# Thêm tất cả các node
workflow.add_node("router", router.proxy_router_node)
workflow.add_node("manager", manager.manager_node)
workflow.add_node("booking_agent", booking.booking_node)
workflow.add_node("cancel_booking_agent", cancel_booking.cancel_booking_node)
workflow.add_node("general_agent", general.general_node)
workflow.add_node("tools", tool_node)

# 3. Định nghĩa các cạnh (giữ nguyên code cũ của bạn)
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["next_agent"],
    {
        "manager": "manager",
        "booking_agent": "booking_agent",
        "cancel_booking_agent": "cancel_booking_agent",
        "general_agent": "general_agent",
    }
)

def route_from_manager(state: AgentState):
    next_agent = state.get("next_agent")
    return next_agent

workflow.add_conditional_edges("manager", route_from_manager)

def route_after_task(state: AgentState):
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    if state.get("next_agent") == "manager":
        return "manager"
    return END

workflow.add_conditional_edges("booking_agent", route_after_task)
workflow.add_conditional_edges("cancel_booking_agent", route_after_task)
workflow.add_conditional_edges("general_agent", route_after_task)

def route_after_tools(state: AgentState) -> str:
    previous_agent = state.get("previous_agent")
    if previous_agent:
        print(f">>> Tool execution finished. Returning to: {previous_agent}")
        return previous_agent
    else:
        return "manager"

workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "booking_agent": "booking_agent",
        "cancel_booking_agent": "cancel_booking_agent",
        "general_agent": "general_agent",
    }
)

# 4. ✅ THÊM CHECKPOINT VÀO ĐÂY

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)



try:
    app.get_graph().draw_mermaid_png(output_file_path="D:\\Project\\-n-T-t-Nghi-p-\\img\\workflow_graph.png")
    print("Đã vẽ sơ đồ workflow ra file: workflow_graph.png")
except Exception as e:
    print(f"Không thể vẽ đồ thị: {e}")