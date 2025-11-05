# app.py (đặt ở thư mục gốc dự án)
import sys
import os
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

# --- Thêm src vào sys.path để import workflow ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from flight_booking_agent.graph.workflow import app as langgraph_app
except ImportError as e:
    print("Không thể import workflow LangGraph:", e)
    sys.exit(1)


@cl.on_chat_start
async def start_chat():
    """
    Khi user bắt đầu cuộc trò chuyện mới:
    - Khởi tạo một thread_id mới
    - Khởi tạo conversation_state rỗng
    """
    thread_id = f"thread_{cl.user_session.id}"  # mỗi user một thread_id
    cl.user_session.set("conversation_state", {"messages": []})
    cl.user_session.set("current_thread_id", thread_id)

    await cl.Message(
        content="Xin chào! Tôi là FlyAgent, trợ lý ảo chuyên đặt vé máy bay.\nBạn muốn bay từ đâu tới đâu?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Xử lý mỗi tin nhắn của user:
    - Lấy conversation_state theo thread_id
    - Thêm tin nhắn user
    - Stream kết quả AI
    - Cập nhật state theo thread_id
    """
    # Lấy thread_id hiện tại
    thread_id = cl.user_session.get("current_thread_id")
    conversation_state = cl.user_session.get("conversation_state")

    # Thêm tin nhắn user vào state
    conversation_state["messages"].append(HumanMessage(content=message.content))

    # Tạo tin nhắn trống để stream AI
    response_message = cl.Message(content="", parent=message.id)
    await response_message.send()

    try:
        # Stream các event từ LangGraph
        async_streamer = cl.make_async(langgraph_app.stream)
        final_state = None

        async for event in await async_streamer(conversation_state, thread_id=thread_id):
            for node_name, node_output in event.items():
                # Lấy tin nhắn mới từ node
                new_messages = node_output.get("messages", [])
                if new_messages:
                    last_message = new_messages[-1]
                    if isinstance(last_message, AIMessage) and last_message.content:
                        await response_message.stream_token(last_message.content)
            final_state = event

        # Lưu lại state cuối cùng cho thread_id
        if final_state:
            # Lấy state từ node cuối cùng
            last_state = list(final_state.values())[-1]
            cl.user_session.set("conversation_state", last_state)

        # Hoàn tất tin nhắn
        await response_message.update()

    except Exception as e:
        await cl.Message(content=f"Đã xảy ra lỗi: {e}").send()
        cl.user_session.set("conversation_state", {"messages": []})
