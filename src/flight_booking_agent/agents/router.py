from ..graph.state import AgentState

SPECIALIST_AGENTS = ["booking_agent", "cancel_booking_agent", "general_agent"]

def proxy_router_node(state: AgentState) -> dict:
    previous_agent = state.get("previous_agent")

    # Nếu chưa từng gọi agent nào → đi qua manager
    if previous_agent is None:
        return {"next_agent": "manager"}

    # Nếu đã có agent và agent đó thuộc nhóm chuyên trách → quay lại đúng agent đó
    if previous_agent in SPECIALIST_AGENTS:
        return {"next_agent": previous_agent}

    # Mặc định → vẫn về manager
    return {"next_agent": "manager"}