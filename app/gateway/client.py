import logfire 
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL 
from langchain_openai import ChatOpenAI

from app.config import settings

# Production gateway config:
#  - Fallback: primary @rag/llama-3.3-70b-versatile -> @brag/llama-3.1-8b-instant on failure
# -Cache : semantic mode (requires Portkey Enterprise - silently falls back to simple on free/starter)
#  - Retry : 2 attempts on rate limit / server error before triggering the fallback target

GATEWAY_CONFIG = {
    "strategy": {"mode":"fallback"},
    "cache":{"mode":"simple"},
    "retry":{
        "attempts":2,
        "on_status_codes": [429, 503]
    },
    "targets":[
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/allam-2-7b"}},
        {"override_params":{"model":f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"}}
    ]
}

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY
)

from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

def get_langchain_llm(feature: str="rag"):
    """
    Returns a Portkey-backed ChatOpenAI with automatic fallback to Gemini
    when rate limits (429) or transient errors occur.
    """
    primary_llm = ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        max_retries=1,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            metadata={
                "feature": feature,
                "_user":"rag-system",
                "environmnent":"production"
            }
        )
    )

    if settings.GEMINI_API_KEY:
        gemini_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0
        )
        return primary_llm.with_fallbacks([gemini_llm])

    return primary_llm

def generate_gemini_fallback(prompt: str) -> str:
    """
    Direct fallback to Google Gemini when Portkey / Groq hits a rate limit or fails.
    Defensively tries primary GEMINI_MODEL followed by gemini-3.5-flash if high demand occurs.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    candidate_models = [settings.GEMINI_MODEL, "gemini-3.5-flash"]
    last_err = None

    for model_name in candidate_models:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if res.text:
                return res.text.strip()
        except Exception as e:
            last_err = e
            logfire.warning(f"⚠️ Gemini model {model_name} failed: {e}. Trying next candidate...")
            continue

    if last_err:
        raise last_err
    return ""

def extract_cache_status(response)->str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status= getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"