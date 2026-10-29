from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire
llm = get_langchain_llm(feature="planner")

OFF_TOPIC_REFUSAL = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
    "I can't help with that — but ask me anything technical!"
)

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed, if the query is conversational,
    or if the query is off-topic (e.g. coffee, cooking, general knowledge).
    """
    # Get the conversation history (excluding the latest message)
    history=""
    for msg in state.get("messages", [])[:-1]:
        role = "User" if msg["role"]=="user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state.get("messages") else ""

    prompt = f"""
    You are an intelligent Assistant Planner for an Enterprise IT Assistant.
    The assistant is strictly specialized in:
    - Kubernetes (pods, deployments, kubectl CLI, autoscaling, jobs, cronjobs, redis work queues, metrics-server, architecture)
    - Databricks and batch job workflows
    - Intel Hardware (CPUs, FPGAs, NICs, SRIOV)
    - Enterprise Networking (SDN, VLANs, BGP, routing)

    Analyze the conversation history and the latest user message:

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Classification rules:
    1. Respond with 'CONVERSATIONAL' ONLY if the message is purely a social greeting or pleasantry (e.g., "hi", "hello", "thanks", "who are you") OR asks about prior chat memory (e.g., "what did I just say?"). NEVER classify any query containing technical terms or IT concepts as CONVERSATIONAL.
    2. If the message asks ANY technical question, definition, tool explanation, or how-to (e.g. "what is kubectl?", "how to scale pods?", "how to start redis?"), it is strictly TECHNICAL. You MUST output a concise refined search query.
    3. If it is an OFF-TOPIC question completely outside enterprise IT (e.g., how to make coffee, cooking, recipes, jokes, sports, pop culture, weather), respond with 'OFF_TOPIC'.

    Output ONLY 'CONVERSATIONAL', 'OFF_TOPIC', or the refined search query.
    """

    print(f"\n📥 [PLANNER] Received query: '{user_message}'")

    with logfire.span("🧠 Planner Decision"):
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            decision = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
        else:
            decision = str(content).strip()
        print(f"🧠 [PLANNER] Raw LLM Decision: '{decision}'")
        logfire.info(f"Intent identified: {decision}")

    upper_dec = decision.upper()
    if upper_dec == "OFF_TOPIC" or upper_dec.startswith("OFF_TOPIC"):
        print(f"🚫 [PLANNER] Decision: OFF_TOPIC -> Skipping Vector DB (Blocked)\n")
        logfire.info(f"🛡️ Planner blocked off-topic query | query='{user_message[:80]}'")
        return {
            "current_query": "OFF_TOPIC",
            "status": "Blocked off-topic query.",
            "plan": state.get("plan", []) + ["Intent: Off-Topic Blocked", "Retrieval: Skipped"],
            "final_answer": OFF_TOPIC_REFUSAL,
            "messages": [{"role": "assistant", "content": OFF_TOPIC_REFUSAL}]
        }

    if upper_dec == "CONVERSATIONAL" or upper_dec.startswith("CONVERSATIONAL"):
        print(f"💬 [PLANNER] Decision: CONVERSATIONAL -> Skipping Vector DB (Using conversation memory)\n")
        return {
            "current_query":"CONVERSATIONAL",
            "status":"Handling conversationally (using memory)...",
            "plan":["Intent: Conversationa;/Memory",           "Retrieval: Skipped"]
        }
    
    # Extract clean search query if model returned verbose formatting
    search_query = decision
    if "Refined search query:" in search_query:
        search_query = search_query.split("Refined search query:")[-1].strip().strip('"').strip("'")
    elif "search query:" in search_query.lower():
        idx = search_query.lower().find("search query:")
        search_query = search_query[idx + len("search query:"):].strip().strip('"').strip("'")

    print(f"⏩ [PLANNER] Decision: TECHNICAL -> Routing to Vector DB with query: '{search_query}'\n")

    return {
        "current_query": search_query,
        "status": f"Technical research needed. Searching for: {search_query}",
        "plan": ["Intent:Technical", f"Search Term: {search_query}"]
    }
