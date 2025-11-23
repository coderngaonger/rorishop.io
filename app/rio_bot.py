# app/rio_bot.py
import os
from typing import List, Tuple

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

from .settings import configure_llama_index

PERSIST_DIR = os.getenv("PERSIST_DIR", "/Users/minhhieu/Documents/rioshoop/db")


# ====== System prompt cho RIO BOT ======
RIO_SYSTEM_PROMPT = """
Bạn là RIO BOT, trợ lý tư vấn khách hàng cho shop RIO.

Nhiệm vụ:
- Giải thích, tư vấn về sản phẩm, loại hàng, kích cỡ, chất liệu, giá, khuyến mãi,
  chính sách giao hàng và đổi trả của shop RIO.
- Ưu tiên dùng tiếng Việt, văn phong thân thiện, ngắn gọn, dễ hiểu.
- Khi trả lời các câu hỏi liên quan đến sản phẩm/chính sách, phải ưu tiên sử dụng
  thông tin lấy được từ tài liệu (RAG) đã cung cấp cho bạn.
- Nếu tài liệu không chứa thông tin cần thiết, hãy nói rõ là bạn không chắc
  và gợi ý khách liên hệ nhân viên RIO để được hỗ trợ thêm.
- Không tự bịa đặt thông tin về giá, tồn kho, hoặc chương trình khuyến mãi.

Bạn chỉ là trợ lý ảo của shop RIO, không phải con người.
"""


class RioChatbot:
    def __init__(self) -> None:
        # Cấu hình LLM + embedding
        configure_llama_index()

        # Load index từ storage
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        self.index = load_index_from_storage(storage_context)

        # Short memory cho hội thoại
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)

        # Tạo chat_engine dùng RAG + memory + prompt
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=self.memory,
            similarity_top_k=5,
            system_prompt=RIO_SYSTEM_PROMPT,
        )

    @staticmethod
    def wrapper_chat_history(messages: List[ChatMessage]) -> list[str]:
        """
        Chuyển chat_history (List[ChatMessage]) -> list[str] đơn giản.
        Dùng để log hoặc trả ra cho UI.
        """
        history: list[str] = []
        for m in messages:
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.content is not None:
                history.append(m.content)
        return history

    def converse(self, message: str) -> Tuple[str, list[str]]:
        """
        Gửi 1 message vào chat_engine, trả về:
          - answer: nội dung trả lời (string)
          - short_history: list[str] lịch sử hội thoại rút gọn
        """
        response = self.chat_engine.chat(message)

        full_history = self.chat_engine.chat_history
        short_history = self.wrapper_chat_history(full_history[-8:])

        return response.response, short_history


# CLI đơn giản để test trong container
if __name__ == "__main__":
    bot = RioChatbot()
    print("RIO BOT sẵn sàng. Gõ 'exit' để thoát.")

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in {"exit", "quit"}:
            break

        answer, _ = bot.converse(user_input)
        print("RIO BOT:", answer)
