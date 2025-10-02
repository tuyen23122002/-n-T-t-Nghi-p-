
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from langchain_core.messages import HumanMessage
from src.flight_booking_agent.graph.workflow import app

# Chúng ta sẽ lưu trữ lịch sử tin nhắn ở đây
# Trong ứng dụng thực tế, bạn sẽ lưu nó vào database
conversation_history = []

if __name__ == "__main__":
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
        result = app.invoke(inputs)

        # Lấy tin nhắn cuối cùng (là của AI) từ kết quả
        ai_message = result["messages"][-1]

        # Thêm tin nhắn của AI vào lịch sử
        conversation_history.append(ai_message)

        # In ra câu trả lời của AI
        print("Chatbot:", ai_message.content)