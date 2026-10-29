import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway import portkey_client, extract_cache_status, generate_gemini_fallback

def generate_node(state:AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]
    history_str=""
    for msg in state.get("messages", [])[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state.get("messages") else ""

    if query == "CONVERSATIONAL":
        print("💬 [RESPONDER] Synthesizing conversational response from memory (Vector DB context NOT used)...")
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
            You are a professional Enterprise IT Assistant strictly focused on Kubernetes, Intel hardware, and Enterprise Networking.
            Answer the user's latest message using the CONVERSATION HISTORY below.
            If the user asks an off-topic question unrelated to Enterprise IT (such as cooking, recipes, coffee, sports, entertainment, or general trivia), politely refuse by stating:
            "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"

            CONVERSATION HISTORY:
            {history_str}

            LATEST MESSAGE:
            "{user_msg}"
        """
    else:
        doc_count = len(state.get("documents", []))
        print(f"📚 [RESPONDER] Synthesizing technical response using {doc_count} Vector DB context chunks...")
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.warning(f"⚠️ Primary LLM Generation failed: {e}. Falling back to Gemini...")
            if settings.GEMINI_API_KEY:
                try:
                    content = generate_gemini_fallback(prompt)
                    logfire.info("✅ Response successfully synthesised via Gemini fallback.")
                    return {
                        "final_answer": content,
                        "status": "Response generated (Gemini fallback).",
                        "plan": state["plan"] + ["Fallback: Gemini"],
                        "messages": [{"role": "assistant", "content": content}]
                    }
                except Exception as gemini_err:
                    logfire.error(f"Gemini fallback failed: {gemini_err}")
                    raise gemini_err
            raise e
