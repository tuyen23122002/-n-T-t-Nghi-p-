from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from ..graph.state import AgentState
from ..config import llm

# Pydantic model để định nghĩa output có cấu trúc cho Manager
class ManagerHandoff(BaseModel):
    target_agent_name: Literal["booking_agent", "general_agent", "cancel_booking_agent", "END"] = Field(
        ..., description="Agent chuyên trách phù hợp để xử lý yêu cầu."
    )

def manager_node(state: AgentState) -> dict:
    """
    Node điều phối: Phân tích yêu cầu và quyết định agent tiếp theo.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """"## Vai trò ##"
            "Bạn là **Vivi**, là một trợ lý ảo đóng vai trò điều phối viên, "
            "Nhiệm vụ **DUY NHẤT** của bạn là phân tích yêu cầu của người dùng và "
            "**chuyển hướng chính xác** đến agent chuyên trách phù hợp. "
            "Bạn **TUYỆT ĐỐI KHÔNG** được tự mình trả lời bất kỳ câu hỏi nào của khách hàng.\n\n"

            "## Các agent chuyên trách & phạm vi xử lý ##\n"
            "Để phân loại chính xác, hãy nắm rõ phạm vi xử lý của từng agent:\n"
            "1. `booking_agent`: xử lý các yêu cầu **LIÊN QUAN TRỰC TIẾP** đến việc **TÌM KIẾM, LỰA CHỌN, ĐẶT MUA** vé máy bay và "
            "các dịch vụ đi kèm (hành lý, suất ăn, chọn chỗ) cho một chuyển bay cụ thể mà khách hàng đang quan tâm hoặc muốn đặt.\n"
            "2. `cancel_booking_agent`: xử lý các yêu cầu **LIÊN QUAN TRỰC TIẾP** đến việc hủy bỏ một vé đã đặt. "
            "Yêu cầu này thường bao gồm mã đặt chỗ hoặc thông tin đủ để xác định booking cần hủy.\n"
            "3. `general_agent`: xử lý các yêu cầu khác nằm ngoài phạm vi của các agent trên.\n\n"

            "## Quy tắc phân loại và chuyển hướng ##"
            "Phân tích **ý định** của người dùng một cách cẩn thận và áp dụng quy tắc sau để xác định tác tử đích:\n"
            "1. Nếu yêu cầu rõ ràng là muốn **HỦY** vé -> chuyển hướng đến `cancel_booking_agent`.\n"
            "2. Nếu yêu cầu rõ ràng là muốn **TÌM, CHỌN, ĐẶT MUA**` một chuyến bay hoặc dịch vụ đi kèm -> "
            "chuyển hướng đến `booking_agent`.\n"
            "3. Trong **TẤT CẢ các trường hợp còn lại** (bao gồm hỏi thông tin chung, quy định, thủ tục, chào hỏi, "
            "các yêu cầu không rõ ràng hoặc không liên quan đến việc đặt/hủy cụ thể) -> Chuyển hướng đến `general_agent`"

            "## Lưu ý quan trọng ##\n"
            "Bạn **PHẢI LUÔN** phản hồi dưới dạng cấu trúc `ManagerHandoff`"
        )"""),
            ("user", "Yêu cầu của người dùng: {input}"),
        ]
    )

    # Sử dụng .with_structured_output để đảm bảo LLM trả về đúng format
    structured_llm = llm.with_structured_output(ManagerHandoff)
    
    chain = prompt | structured_llm
    
    # Chỉ lấy tin nhắn cuối cùng của người dùng để phân tích
    user_input = state['messages'][-1].content
    result = chain.invoke({"input": user_input})
    
    return {"next_agent": result.target_agent_name}