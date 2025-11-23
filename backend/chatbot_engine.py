import os
from typing import List, Tuple
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

load_dotenv()

# System prompt cho RIO BOT
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


class ChatbotEngine:
    def __init__(self, persist_dir: str):
        """Khởi tạo chatbot engine với RAG index đã được persist."""
        self.persist_dir = persist_dir
        self._setup_llm()
        self._setup_embeddings()
        self._load_index()
        self._setup_chat_engine()

    def _setup_llm(self):
        """Cấu hình LLM (Gemini)."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        llm = Gemini(
            api_key=api_key,
            model="models/gemini-2.5-flash",
            temperature=0.1
        )
        Settings.llm = llm

    def _setup_embeddings(self):
        """Cấu hình embedding model."""
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        )

    def _load_index(self):
        """Load index từ persist directory."""
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self.index = load_index_from_storage(storage_context)

    def _setup_chat_engine(self):
        """Thiết lập chat engine với memory."""
        memory = ChatMemoryBuffer.from_defaults(token_limit=1024)
        
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=5,
            system_prompt=RIO_SYSTEM_PROMPT,
        )

    def _wrapper_chat_history(self, messages: List[ChatMessage]) -> List[str]:
        """Chuyển chat_history thành list string."""
        history = []
        for m in messages:
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.content:
                history.append(m.content)
        return history

    def chat(self, message: str) -> Tuple[str, List[str]]:
        """
        Gửi message và nhận response.
        
        Returns:
            Tuple[str, List[str]]: (answer, short_history)
        """
        response = self.chat_engine.chat(message)
        
        # Lấy lịch sử hội thoại (8 message gần nhất)
        full_history = self.chat_engine.chat_history
        short_history = self._wrapper_chat_history(full_history[-8:])
        
        return response.response, short_history

    def reset_chat(self):
        """Reset lịch sử hội thoại."""
        self._setup_chat_engine()