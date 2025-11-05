from ..graph.state import AgentState

SPECIALIST_AGENTS = ["booking_agent", "cancel_booking_agent", "general_agent"]

def proxy_router_node(state: AgentState) -> dict:
    """
    Node này hoạt động như một UserProxy. Nó quyết định xem có nên
    gửi thẳng yêu cầu đến agent trước đó hay không, hay phải qua Manager.
    """

    previous_agent = state.get("previous_agent")
    
    if previous_agent and previous_agent in SPECIALIST_AGENTS:
        return {"next_agent": previous_agent}
    else:
        return {"next_agent": "manager"}