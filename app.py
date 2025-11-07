import chainlit as cl
import httpx
import uuid

# URL của FastAPI backend
BASE_URL = "http://127.0.0.1:8001"  # Thay đổi nếu FastAPI của bạn chạy trên một địa chỉ khác

@cl.on_chat_start
async def on_chat_start():
    """
    Hàm này được gọi khi một cuộc trò chuyện mới bắt đầu.
    Nó tạo ra một thread_id duy nhất cho mỗi cuộc trò chuyện.
    """
    # Tạo một thread_id mới cho mỗi phiên trò chuyện
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        content=f"Xin chào! 👋 Mình là FlyAgent – trợ lý đặt vé máy bay của bạn.",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Hàm này được gọi mỗi khi người dùng gửi một tin nhắn.
    Nó gửi tin nhắn đến API FastAPI và hiển thị phản hồi.
    """
    thread_id = cl.user_session.get("thread_id")

    if not thread_id:
        # Xử lý trường hợp không tìm thấy thread_id
        await cl.Message(
            content="Đã xảy ra lỗi: không tìm thấy thread_id. Vui lòng thử làm mới trang."
        ).send()
        return

    # Dữ liệu để gửi đến API
    chat_request = {
        "message": message.content,
        "thread_id": thread_id
    }

    async with httpx.AsyncClient() as client:
        try:
            # Gửi yêu cầu POST đến endpoint /chat
            response = await client.post(f"{BASE_URL}/chat", json=chat_request, timeout=30.0)
            response.raise_for_status()  # Ném ra một ngoại lệ nếu có lỗi HTTP

            chat_response = response.json()

            # Gửi phản hồi của bot trở lại giao diện người dùng
            await cl.Message(
                content=chat_response.get("response", "Không nhận được phản hồi hợp lệ từ bot."),
            ).send()

        except httpx.HTTPStatusError as e:
            await cl.Message(
                content=f"Đã xảy ra lỗi khi giao tiếp với bot: {e.response.status_code} - {e.response.text}",
            ).send()
        except httpx.RequestError as e:
            await cl.Message(
                content=f"Đã xảy ra lỗi mạng: {e}",
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"Đã xảy ra một lỗi không mong muốn: {e}",
            ).send()