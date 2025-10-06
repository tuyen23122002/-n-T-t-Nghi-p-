# main.py
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Thêm src vào sys.path để có thể import từ đó
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.flight_booking_agent.graph.workflow import app

def run_conversation():
    # Sử dụng một list để lưu trữ lịch sử tin nhắn
    conversation_history: list[BaseMessage] = []

    print("Chatbot đã sẵn sàng. Gõ 'thoát' để kết thúc.")
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ["thoát", "exit"]:
            print("Chatbot: Tạm biệt!")
            break

        # Thêm tin nhắn của người dùng vào lịch sử
        conversation_history.append(HumanMessage(content=user_input))

        # Tạo đầu vào cho đồ thị
        inputs = {"messages": conversation_history}
        
        # Chạy đồ thị
        # result chính là trạng thái cuối cùng của đồ thị sau khi chạy
        result = app.invoke(inputs)

        # Lấy danh sách tin nhắn từ kết quả
        final_messages = result["messages"]

        # Tìm tin nhắn cuối cùng là của AI để hiển thị
        # Cách làm này an toàn hơn việc chỉ lấy phần tử cuối cùng
        last_ai_message = None
        if final_messages and isinstance(final_messages[-1], AIMessage):
            last_ai_message = final_messages[-1]
        
        if last_ai_message:
            print("Chatbot:", last_ai_message.content)
            # Cập nhật lịch sử với toàn bộ tin nhắn mới
            conversation_history = final_messages
        else:
            # Trường hợp này xảy ra nếu đồ thị kết thúc mà không tạo ra tin nhắn AI mới
            # (ví dụ: đang chờ gọi tool)
            print("Chatbot: [Đang xử lý...]")
            # Vẫn cập nhật lịch sử để giữ trạng thái
            conversation_history = final_messages

if __name__ == "__main__":
    run_conversation()