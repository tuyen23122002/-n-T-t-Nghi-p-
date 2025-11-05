# main.py
import sys
import os
import uuid  # Import thư viện để tạo ID duy nhất
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Thêm src vào sys.path để có thể import từ đó
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.flight_booking_agent.graph.workflow import app

def run_conversation():
    # 1. TẠO CONFIG CHO LUỒNG HỘI THOẠI (THREAD)
    # Mỗi luồng sẽ có state riêng được LangGraph tự động lưu trữ.
    # Trong ứng dụng thực tế, mỗi người dùng/phiên chat sẽ có một ID riêng.
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    print(f"Chatbot đã sẵn sàng. Phiên của bạn là: {session_id}. Gõ 'thoát' để kết thúc.")
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ["thoát", "exit"]:
            print("Chatbot: Tạm biệt!")
            break

        # 2. CHỈ CẦN CUNG CẤP TIN NHẮN MỚI
        # Không cần tự quản lý lịch sử hội thoại nữa.
        # LangGraph sẽ tự động nạp lịch sử cũ từ thread_id và thêm tin nhắn mới này vào.
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # 3. CHẠY ĐỒ THỊ VỚI INPUTS VÀ CONFIG
        # `config` sẽ cho LangGraph biết phải làm việc trên state của luồng nào.
        # Biến `result` sẽ chứa state cuối cùng SAU KHI chạy xong lượt này.
        result = app.invoke(inputs, config)

        # 4. XỬ LÝ KẾT QUẢ (KHÔNG CẦN CẬP NHẬT LỊCH SỬ THỦ CÔNG)
        # Lấy danh sách tin nhắn từ kết quả cuối cùng
        final_messages = result["messages"]

        # Tìm và in ra tin nhắn cuối cùng của AI
        last_ai_message = None
        if final_messages and isinstance(final_messages[-1], AIMessage) and final_messages[-1].content:
            last_ai_message = final_messages[-1]
        
        if last_ai_message:
            print("Chatbot:", last_ai_message.content)
        else:
            # Trường hợp này hiếm khi xảy ra, có thể là khi agent chỉ gọi tool
            # mà không trả lời gì, hoặc có lỗi.
            print("Chatbot: ... (đang xử lý)")

if __name__ == "__main__":
    run_conversation()