import os, csv, json, random
from datetime import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lyft Driver Onboarding — Multi-Agent AI",
    page_icon="🚗",
    layout="centered",
)

st.markdown("""
<style>
.agent-badge {
    display: inline-block;
    background: #FF00BF;
    color: white;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.trace-box {
    background: #1a1a2e;
    color: #a0f0b0;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.82rem;
    font-family: monospace;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── API key ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ── Data loading (cached) ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_resource
def load_data():
    def load_csv(name):
        with open(os.path.join(DATA_DIR, name)) as f:
            return list(csv.DictReader(f))

    def load_json(name):
        with open(os.path.join(DATA_DIR, name)) as f:
            return json.load(f)

    drivers_raw           = load_csv("drivers.csv")
    documents_raw         = load_csv("documents.csv")
    background_checks_raw = load_csv("background_checks.csv")
    inspection_slots_raw  = load_csv("inspection_slots.csv")
    support_tickets_raw   = load_csv("support_tickets.csv")
    policy_kb             = load_json("policy_kb.json")

    driver_db = {}
    for row in drivers_raw:
        driver_db[row["driver_id"]] = {
            "name":  row["name"],
            "city":  row["city"],
            "stage": row["application_stage"],
            "signup_date": row["signup_date"],
            "documents": {},
            "background_check": "not_started",
            "background_check_flag_reason": "",
            "inspection": "not_scheduled",
        }
    for row in documents_raw:
        d = driver_db.get(row["driver_id"])
        if d:
            d["documents"][row["doc_type"]] = row["status"]
    for row in background_checks_raw:
        d = driver_db.get(row["driver_id"])
        if d:
            d["background_check"] = row["status"]
            d["background_check_flag_reason"] = row.get("flag_reason", "")
    for row in inspection_slots_raw:
        if row["status"] == "booked" and row["driver_id"] in driver_db:
            driver_db[row["driver_id"]]["inspection"] = (
                f"scheduled: {row['date']} {row['time']} at {row['center_id']}"
            )

    tickets_by_driver = {}
    for row in support_tickets_raw:
        tickets_by_driver.setdefault(row["driver_id"], []).append(row)

    return driver_db, drivers_raw, policy_kb, tickets_by_driver

DRIVER_DB, drivers_raw, POLICY_KB_ENTRIES, SUPPORT_TICKETS_BY_DRIVER = load_data()

# ── LLM (cached) ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_agents():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    # ── Tools: Agent 1 ─────────────────────────────────────────────────────────
    @tool
    def check_documents(driver_id: str) -> str:
        """Check the status of a driver's uploaded documents (license, insurance, registration)."""
        driver = DRIVER_DB.get(driver_id)
        if not driver:
            return "No driver found with id " + driver_id
        docs = driver["documents"]
        missing = [d for d, s in docs.items() if s != "uploaded"]
        if not missing:
            return "All documents uploaded and valid for " + driver["name"] + "."
        details = ", ".join(f"{d} ({docs[d]})" for d in missing)
        return driver["name"] + " needs attention on: " + details + "."

    @tool
    def run_background_check(driver_id: str) -> str:
        """Start or check the status of a driver's background check.
        Returns one of: not_started, in_progress, cleared, flagged."""
        driver = DRIVER_DB.get(driver_id)
        if not driver:
            return "No driver found with id " + driver_id
        status = driver["background_check"]
        if status == "not_started":
            driver["background_check"] = "in_progress"
            return "Background check started for " + driver["name"] + ". Typically takes 2-3 business days."
        if status == "flagged":
            return (
                "Background check for " + driver["name"] + " was FLAGGED ("
                + str(driver["background_check_flag_reason"])
                + ") and needs human review."
            )
        return "Background check for " + driver["name"] + " is currently: " + status + "."

    @tool
    def schedule_inspection(driver_id: str, preferred_date: str) -> str:
        """Schedule a vehicle inspection for a driver at a preferred date.
        Only allowed once the background check is cleared."""
        driver = DRIVER_DB.get(driver_id)
        if not driver:
            return "No driver found with id " + driver_id
        if driver["background_check"] != "cleared":
            return (
                "Cannot schedule inspection yet — background check is not cleared ("
                + driver["background_check"] + ")."
            )
        slot = preferred_date + " at " + random.choice(["9:00 AM", "11:30 AM", "2:00 PM"])
        driver["inspection"] = "scheduled: " + slot
        return "Inspection scheduled for " + driver["name"] + " on " + slot + " at the nearest Lyft Hub."

    @tool
    def lookup_policy(question: str) -> str:
        """Look up an answer to a driver onboarding policy question."""
        q_words = set(question.lower().replace("?", "").split())
        best, best_score = None, 0
        for entry in POLICY_KB_ENTRIES:
            haystack = (entry["topic"] + " " + entry["question"]).lower()
            score = len(q_words & set(haystack.replace("?", "").split()))
            if score > best_score:
                best, best_score = entry, score
        return best["answer"] if best else "No policy match found — escalate to a human."

    @tool
    def create_escalation_ticket(driver_id: str, issue_summary: str) -> str:
        """Create a human support ticket when the driver's issue can't be resolved automatically."""
        driver = DRIVER_DB.get(driver_id, {})
        name = driver.get("name", driver_id)
        ticket_id = "TCK-" + str(random.randint(10000, 99999))
        prior = SUPPORT_TICKETS_BY_DRIVER.get(driver_id, [])
        note = f" (driver has {len(prior)} prior ticket(s) on file)" if prior else ""
        return (
            "Escalation ticket " + ticket_id + " created for " + name + note + ". "
            "Summary: " + issue_summary + ". A human specialist will follow up within 24 hours."
        )

    onboarding_assistant = create_react_agent(
        llm,
        tools=[check_documents, run_background_check, schedule_inspection,
               lookup_policy, create_escalation_ticket],
        prompt=(
            "You are Lyft's Driver Onboarding Assistant. You help new drivers get from "
            "signup to their first ride as smoothly as possible.\n\n"
            "You can: check document status, start/check background checks, schedule vehicle "
            "inspections (only once background check is cleared), answer policy questions, "
            "and escalate to a human when needed.\n\n"
            "Always use the driver_id given to you when calling tools. Be warm, clear, and "
            "concise. If a driver's background check is flagged, escalate immediately."
        )
    )

    # ── Tools: Agent 2 ─────────────────────────────────────────────────────────
    TODAY = datetime(2026, 7, 10)

    @tool
    def get_stalled_drivers() -> str:
        """Find drivers stuck in onboarding for more than 5 days. Returns the 5 most-stalled."""
        stalled = []
        for row in drivers_raw:
            did   = row["driver_id"]
            stage = row["application_stage"]
            if stage == "active_driver":
                continue
            signup = datetime.strptime(row["signup_date"], "%Y-%m-%d")
            days   = (TODAY - signup).days
            if days > 5:
                stalled.append(
                    (days, f"{did} ({row['name']}) — stage: {stage}, {days} days since signup")
                )
        if not stalled:
            return "No stalled drivers found."
        stalled.sort(reverse=True)
        return "\n".join(line for _, line in stalled[:5])

    @tool
    def send_nudge(driver_id: str, message: str) -> str:
        """Send a proactive nudge message to a driver (simulated)."""
        name = DRIVER_DB.get(driver_id, {}).get("name", driver_id)
        return f'Nudge sent to {name} ({driver_id}): "{message}"'

    nudge_agent = create_react_agent(
        llm,
        tools=[get_stalled_drivers, send_nudge],
        prompt=(
            "You are Lyft's Proactive Onboarding Nudge Agent. Find stalled drivers using "
            "get_stalled_drivers, then for EACH one write a short warm nudge mentioning their "
            "exact stage and send it with send_nudge. End with a one-line summary."
        )
    )

    # ── Tools: Agent 3 ─────────────────────────────────────────────────────────
    @tool
    def check_vehicle_eligibility(make: str, model: str, year: int, city: str) -> str:
        """Check whether a vehicle qualifies to drive for Lyft based on model year."""
        min_year = 2012
        if year >= min_year:
            return f"{year} {make} {model} is eligible to drive for Lyft in {city} (minimum year: {min_year})."
        return (
            f"{year} {make} {model} does NOT meet the minimum vehicle year "
            f"requirement ({min_year}+) for {city}."
        )

    eligibility_agent = create_react_agent(
        llm,
        tools=[check_vehicle_eligibility, lookup_policy],
        prompt=(
            "You are Lyft's Vehicle Eligibility Pre-Check Agent for prospective drivers. "
            "Use check_vehicle_eligibility to verify the car qualifies. "
            "Use lookup_policy for any other questions. Be encouraging and tell them what to do next."
        )
    )

    return onboarding_assistant, nudge_agent, eligibility_agent

# ── Helper ─────────────────────────────────────────────────────────────────────
def extract_text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)

def run_agent(agent, message: str):
    tool_trace = []
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_trace.append(f"🔧 {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            tool_trace.append(f"   ↩ {msg.content[:180]}")
    reply = extract_text_content(result["messages"][-1].content)
    return reply, "\n".join(tool_trace)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🚗 Lyft Driver Onboarding")
st.caption("Multi-Agent AI System · LangGraph + Gemini · Capstone Project")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Agent 1 — Onboarding Chat",
    "📣 Agent 2 — Nudge Agent",
    "✅ Agent 3 — Eligibility Check",
    "ℹ️ About",
])

onboarding_assistant, nudge_agent, eligibility_agent = load_agents()

DRIVER_CHOICES = {
    f"{row['driver_id']} — {row['name']} ({row['city']}, {row['application_stage']})": row["driver_id"]
    for row in drivers_raw
}

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<span class='agent-badge'>REACTIVE · ReAct</span>", unsafe_allow_html=True)
    st.markdown(
        "Chat as a specific driver. The agent autonomously picks which tools to call — "
        "document check, background check, inspection scheduling, policy lookup, or escalation."
    )

    selected_label = st.selectbox("Select Driver", list(DRIVER_CHOICES.keys()))
    driver_id = DRIVER_CHOICES[selected_label]

    # Example questions
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    examples = [
        "What's still missing from my application?",
        "How long does the background check take?",
        "Can you schedule my inspection for next Tuesday?",
        "Where do I stand overall?",
        "My insurance doc keeps getting rejected, why?",
        "What documents do I need to upload?",
    ]
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.prefill = ex

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message…", key="chat_input")
    if not user_input and st.session_state.prefill:
        user_input = st.session_state.prefill
        st.session_state.prefill = ""

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking…"):
                task = f'Driver ID: {driver_id}. Driver said: "{user_input}"'
                try:
                    reply, trace = run_agent(onboarding_assistant, task)
                except Exception as e:
                    reply, trace = f"⚠️ Error: {e}", ""

            st.markdown(reply)
            if trace:
                with st.expander("🔍 Agent reasoning trace"):
                    st.markdown(f"<div class='trace-box'>{trace}</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("🗑 Clear chat", key="clear"):
        st.session_state.chat_history = []
        st.rerun()

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<span class='agent-badge'>PROACTIVE · ReAct</span>", unsafe_allow_html=True)
    st.markdown(
        "This agent runs on a schedule (daily), finds the most-stalled drivers, and sends "
        "each a personalised nudge — no driver message triggers it."
    )

    if st.button("▶ Run Today's Nudge Check", type="primary"):
        with st.spinner("Agent scanning drivers and sending nudges…"):
            try:
                summary, trace = run_agent(nudge_agent, "Run today's onboarding nudge check.")
            except Exception as e:
                summary, trace = f"⚠️ Error: {e}", ""

        st.subheader("Summary")
        st.markdown(summary)

        if trace:
            with st.expander("🔍 Agent reasoning trace"):
                st.markdown(f"<div class='trace-box'>{trace}</div>", unsafe_allow_html=True)

# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<span class='agent-badge'>PRE-SIGNUP · ReAct</span>", unsafe_allow_html=True)
    st.markdown(
        "For prospective drivers who haven't signed up yet. Enter vehicle details to check "
        "eligibility — the agent also answers any policy questions."
    )

    col1, col2 = st.columns(2)
    make  = col1.text_input("Vehicle Make",  placeholder="e.g. Honda")
    model = col2.text_input("Vehicle Model", placeholder="e.g. Civic")
    year  = col1.number_input("Vehicle Year", min_value=2000, max_value=2025, value=2015, step=1)
    city  = col2.text_input("City", placeholder="e.g. Chicago")
    extra = st.text_input("Any other questions? (optional)", placeholder="e.g. What documents will I need?")

    st.markdown("**Quick examples:**")
    ecols = st.columns(2)
    elig_examples = [
        ("Honda Civic 2011, Chicago", "Honda", "Civic", 2011, "Chicago", "What documents will I need?"),
        ("Toyota Camry 2018, Atlanta", "Toyota", "Camry", 2018, "Atlanta", "Does Lyft provide insurance?"),
        ("BMW 3 Series 2009, Seattle", "BMW", "3 Series", 2009, "Seattle", ""),
        ("Ford F-150 2015, Phoenix",   "Ford",  "F-150",  2015, "Phoenix", ""),
    ]
    if "elig_prefill" not in st.session_state:
        st.session_state.elig_prefill = None

    for i, (label, *vals) in enumerate(elig_examples):
        if ecols[i % 2].button(label, key=f"eex_{i}", use_container_width=True):
            st.session_state.elig_prefill = vals

    if st.session_state.get("elig_prefill"):
        make, model, year, city, extra = st.session_state.elig_prefill
        st.session_state.elig_prefill = None

    if st.button("Check Eligibility", type="primary"):
        if not make or not model or not city:
            st.warning("Please fill in Make, Model, and City.")
        else:
            question = f"I have a {int(year)} {make} {model} in {city}, can I drive for Lyft?"
            if extra.strip():
                question += " " + extra.strip()

            with st.spinner("Checking eligibility…"):
                try:
                    reply, trace = run_agent(eligibility_agent, question)
                except Exception as e:
                    reply, trace = f"⚠️ Error: {e}", ""

            st.subheader("Result")
            st.markdown(reply)

            if trace:
                with st.expander("🔍 Agent reasoning trace"):
                    st.markdown(f"<div class='trace-box'>{trace}</div>", unsafe_allow_html=True)

# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
## Lyft Driver Onboarding — Multi-Agent AI System
**Capstone Project · Agentic AI for Business Applications**

### Problem
New driver onboarding is slow and support-team heavy, causing driver drop-off before their first ride.

### Three Agents

| Agent | Type | Trigger | Tools |
|-------|------|---------|-------|
| **1. Onboarding Assistant** | Reactive ReAct | Driver sends a message | `check_documents`, `run_background_check`, `schedule_inspection`, `lookup_policy`, `create_escalation_ticket` |
| **2. Proactive Nudge Agent** | Proactive ReAct | Scheduled — no driver message needed | `get_stalled_drivers`, `send_nudge` |
| **3. Vehicle Eligibility Pre-Check** | Pre-signup ReAct | Prospective driver before sign-up | `check_vehicle_eligibility`, `lookup_policy` |

### ReAct Pattern
```
driver message
      ↓
agent reasons → picks tool → calls it → reads result → reasons again
      ↓
final reply
```

### Stack
- **LLM:** Gemini 1.5 Flash via `langchain-google-genai`
- **Agent framework:** LangGraph `create_react_agent`
- **UI:** Streamlit
- **Data:** 40 synthetic drivers across 10 US cities
""")
