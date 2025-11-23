# app/settings.py
import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini


# ====== Biến môi trường (có thể override khi chạy container) ======
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # bắt buộc phải có

HF_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",  # mặc định tiếng Việt
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "models/gemini-2.5-flash",
)


def configure_llama_index() -> None:
    """
    Config chung cho LlamaIndex:
      - Embedding HF local
      - LLM Gemini
    Gọi hàm này ở cả build_index và chatbot.
    """
    # Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=HF_EMBED_MODEL,
    )

    # LLM
    Settings.llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model=GEMINI_MODEL,
    )
