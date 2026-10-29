import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    QDRANT_COLLECTION = "enterprise_rag"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "allam-2-7b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_SLUG= os.getenv("GROQ_SLUG")
    GROQ_SLUG_2= os.getenv("GROQ_SLUG_2")
    PORTKEY_API_KEY= os.getenv("PORTKEY_API_KEY")

settings = Settings()