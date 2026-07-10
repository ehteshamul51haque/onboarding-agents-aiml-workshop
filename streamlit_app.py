import os, csv, json, random
from datetime import datetime
from typing import Literal
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lyft Onboarding — Multi-Agent System",
    page_icon="🚗",
    layout="centered",
)

st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 8px;
}
.badge-supervisor { background:#1a1a2e; color:#fff; }
.badge-onboarding { background:#0066CC; color:#fff; }
.badge-nudge      { background:#FF00BF; color:#fff; }
.badge-eligibility{ background:#00875A; color:#fff; }
.trace-box {
    background: #0d1117;
    color: #7ee787;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.80rem;
    font-family: monospace;
    white-space: pre-wrap;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── API key ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ── Data loading ───────────────────────────────────────────────────────────────
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
            "name": row["name"], "city": row["city"],
            "stage": row["application_stage"],
            "signup_date": row["signup_date"],
            "documents": {}, "background_check": "not_started",
            "background_check_flag_reason": "", "inspection": "not_scheduled",
        }
    for row in documents_raw:
        d = driver_db.get(row["driver_id"])
        if d: d["documents"][row["doc_type"]] = row["status"]
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

    tickets = {}
    for row in support_tickets_raw:
        tickets.setdefault(row["driver_id"], []).append(row)

    return driver_db, drivers_raw, policy_kb, tickets

DRIVER_DB, drivers_raw, POLICY_KB_ENTRIES, SUPPORT_TICKETS_BY_DRIVER = load_data()

# ── Helper ─────────────────────────────────────────────────────────────────────
def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)

def collect_trace(messages):
    trace = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append(f"🔧 {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            trace.append(f"   ↩ {msg.content[:200]}")
    return "\n".join(trace)

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# ── Agent 1 tools ──────────────────────────────────────────────────────────────
@tool
def check_documents(driver_id: str) -> str:
    """Check the status of a driver's uploaded documents."""
    driver = DRIVER_DB.get(driver_id)
    if not driver: return "No driver found with id " + driver_id
    docs = driver["documents"]
    missing = [d for d, s in docs.items() if s != "uploaded"]
    if not missing: return "All documents uploaded and valid for " + driver["name"] + "."
    return driver["name"] + " needs attention on: " + ", ".join(f"{d} ({docs[d]})" for d in missing) + "."

@tool
def run_background_check(driver_id: str) -> str:
    """Start or check the status of a driver's background check."""
    driver = DRIVER_DB.get(driver_id)
    if not driver: return "No driver found with id " + driver_id
    status = driver["background_check"]
    if status == "not_started":
        driver["background_check"] = "in_progress"
        return "Background check started for " + driver["name"] + ". Typically takes 2-3 business days."
    if status == "flagged":
        return ("Background check for " + driver["name"] + " was FLAGGED ("
                + str(driver["background_check_flag_reason"]) + ") and needs human review.")
    return "Background check for " + driver["name"] + " is currently: " + status + "."

@tool
def schedule_inspection(driver_id: str, preferred_date: str) -> str:
    """Schedule a vehicle inspection. Only allowed once background check is cleared."""
    driver = DRIVER_DB.get(driver_id)
    if not driver: return "No driver found with id " + driver_id
    if driver["background_check"] != "cleared":
        return "Cannot schedule inspection — background check is not cleared (" + driver["background_check"] + ")."
    slot = preferred_date + " at " + random.choice(["9:00 AM", "11:30 AM", "2:00 PM"])
    driver["inspection"] = "scheduled: " + slot
    return "Inspection scheduled for " + driver["name"] + " on " + slot + " at the nearest Lyft Hub."

@tool
def lookup_policy(question: str) -> str:
    """Look up a driver onboarding policy question."""
    q_words = set(question.lower().replace("?", "").split())
    best, best_score = None, 0
    for entry in POLICY_KB_ENTRIES:
        haystack = (entry["topic"] + " " + entry["question"]).lower()
        score = len(q_words & set(haystack.replace("?", "").split()))
        if score > best_score: best, best_score = entry, score
    return best["answer"] if best else "No policy match found."

@tool
def create_escalation_ticket(driver_id: str, issue_summary: str) -> str:
    """Create a human support ticket for unresolvable issues."""
    driver = DRIVER_DB.get(driver_id, {})
    name = driver.get("name", driver_id)
    ticket_id = "TCK-" + str(random.randint(10000, 99999))
    prior = SUPPORT_TICKETS_BY_DRIVER.get(driver_id, [])
    note = f" ({len(prior)} prior ticket(s))" if prior else ""
    return (f"Escalation ticket {ticket_id} created for {name}{note}. "
            f"Summary: {issue_summary}. Specialist follows up within 24 hours.")

# ── Agent 2 tools ──────────────────────────────────────────────────────────────
TODAY = datetime(2026, 7, 10)

@tool
def get_stalled_drivers() -> str:
    """Find the 5 drivers most stuck in onboarding (>5 days since signup, not yet active)."""
    stalled = []
    for row in drivers_raw:
        if row["application_stage"] == "active_driver": continue
        signup = datetime.strptime(row["signup_date"], "%Y-%m-%d")
        days = (TODAY - signup).days
        if days > 5:
            stalled.append((days, row["driver_id"], row["name"], row["application_stage"]))
    if not stalled: return "No stalled drivers found."
    stalled.sort(reverse=True)
    return "\n".join(
        f"{did} ({name}) — stage: {stage}, {days} days since signup"
        for days, did, name, stage in stalled[:5]
    )

@tool
def send_nudge(driver_id: str, message: str) -> str:
    """Send a personalised nudge message to a stalled driver (simulated)."""
    name = DRIVER_DB.get(driver_id, {}).get("name", driver_id)
    return f'Nudge sent to {name} ({driver_id}): "{message}"'

# ── Agent 3 tools ──────────────────────────────────────────────────────────────
@tool
def check_vehicle_eligibility(make: str, model: str, year: int, city: str) -> str:
    """Check if a prospective driver's vehicle meets Lyft's minimum year requirement."""
    min_year = 2012
    if year >= min_year:
        return f"{year} {make} {model} is eligible in {city} (min year: {min_year})."
    return f"{year} {make} {model} does NOT meet the minimum year requirement ({min_year}+) for {city}."

# ══════════════════════════════════════════════════════════════════════════════
# BUILD AGENTS + SUPERVISOR GRAPH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_system():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    # ── Three specialist agents ────────────────────────────────────────────────
    onboarding_agent = create_react_agent(
        llm,
        tools=[check_documents, run_background_check, schedule_inspection,
               lookup_policy, create_escalation_ticket],
        prompt=(
            "You are Lyft's Driver Onboarding Specialist. You help existing drivers "
            "with document status, background checks, inspection scheduling, policy "
            "questions, and escalations. Always use the driver_id provided. "
            "If a background check is flagged, escalate immediately. Be warm and concise."
        )
    )

    nudge_agent = create_react_agent(
        llm,
        tools=[get_stalled_drivers, send_nudge, create_escalation_ticket],
        prompt=(
            "You are Lyft's Proactive Nudge Specialist. Find stalled drivers using "
            "get_stalled_drivers. For each one: write a short warm nudge referencing "
            "their exact stage, send it with send_nudge. If a driver's stage is "
            "'escalated', use create_escalation_ticket instead of sending a nudge. "
            "End with a brief summary of actions taken."
        )
    )

    eligibility_agent = create_react_agent(
        llm,
        tools=[check_vehicle_eligibility, lookup_policy],
        prompt=(
            "You are Lyft's Vehicle Eligibility Specialist for prospective drivers "
            "who have not signed up yet. Use check_vehicle_eligibility first, then "
            "use lookup_policy to answer any follow-up questions about requirements. "
            "Be encouraging and explain what to do next."
        )
    )

    # ── Supervisor ─────────────────────────────────────────────────────────────
    # The supervisor is an LLM that reads the user message and the outputs
    # from subagents, then decides what to do next.

    supervisor_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    SUPERVISOR_PROMPT = """You are the Lyft Onboarding Supervisor. You coordinate three specialist agents:

1. ONBOARDING_AGENT — handles existing drivers: document checks, background checks,
   inspection scheduling, policy questions, escalations. Use when message mentions a
   driver_id (D-XXXX) or asks about an ongoing application.

2. NUDGE_AGENT — proactively finds stalled drivers and sends them personalised reminders.
   Use when asked to run a nudge check, find stalled drivers, or send reminders.

3. ELIGIBILITY_AGENT — checks vehicle eligibility for people not yet signed up.
   Use when someone mentions a car make/model/year and asks if they can drive for Lyft.

Your job:
- Read the user message carefully.
- Decide which agent(s) to call and in what order.
- You may call agents sequentially when the output of one informs the next
  (e.g. find stalled drivers → then escalate flagged ones via onboarding agent).
- Synthesise all agent outputs into a single clear final response.
- Always tell the user which agents you involved and why.

Respond with a JSON object in this exact format:
{
  "reasoning": "brief explanation of your routing decision",
  "agents_to_call": ["ONBOARDING_AGENT"] or ["NUDGE_AGENT"] or ["ELIGIBILITY_AGENT"]
    or ["NUDGE_AGENT", "ONBOARDING_AGENT"] for chained calls,
  "message_for_agent": "the exact message/task to send to the first agent"
}"""

    # ── LangGraph state ────────────────────────────────────────────────────────
    class GraphState(TypedDict):
        user_message: str
        driver_id: str
        supervisor_reasoning: str
        agents_called: list
        agent_outputs: list         # list of (agent_name, reply, trace)
        final_answer: str

    def supervisor_node(state: GraphState) -> GraphState:
        """Supervisor decides which agents to call."""
        msg = state["user_message"]
        if state["driver_id"]:
            msg = f"Driver ID in context: {state['driver_id']}. User said: {msg}"

        response = supervisor_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=msg)
        ])

        raw = extract_text(response.content).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"): raw = raw[:-3]

        try:
            parsed = json.loads(raw)
        except Exception:
            # Fallback: route to onboarding if we can't parse
            parsed = {
                "reasoning": "Could not parse routing decision, defaulting to onboarding agent.",
                "agents_to_call": ["ONBOARDING_AGENT"],
                "message_for_agent": state["user_message"]
            }

        return {
            **state,
            "supervisor_reasoning": parsed.get("reasoning", ""),
            "agents_to_call": parsed.get("agents_to_call", ["ONBOARDING_AGENT"]),
            "agent_outputs": [],
            "final_answer": "",
            "_next_agent_message": parsed.get("message_for_agent", state["user_message"])
        }

    def run_onboarding_node(state: GraphState) -> GraphState:
        msg = state.get("_next_agent_message", state["user_message"])
        if state["driver_id"] and "driver_id" not in msg.lower() and state["driver_id"] not in msg:
            msg = f"Driver ID: {state['driver_id']}. {msg}"
        result = onboarding_agent.invoke({"messages": [HumanMessage(content=msg)]})
        reply = extract_text(result["messages"][-1].content)
        trace = collect_trace(result["messages"])
        outputs = state.get("agent_outputs", []) + [("ONBOARDING_AGENT", reply, trace)]
        return {**state, "agent_outputs": outputs, "_next_agent_message": reply}

    def run_nudge_node(state: GraphState) -> GraphState:
        msg = state.get("_next_agent_message", state["user_message"])
        result = nudge_agent.invoke({"messages": [HumanMessage(content=msg)]})
        reply = extract_text(result["messages"][-1].content)
        trace = collect_trace(result["messages"])
        outputs = state.get("agent_outputs", []) + [("NUDGE_AGENT", reply, trace)]
        return {**state, "agent_outputs": outputs, "_next_agent_message": reply}

    def run_eligibility_node(state: GraphState) -> GraphState:
        msg = state.get("_next_agent_message", state["user_message"])
        result = eligibility_agent.invoke({"messages": [HumanMessage(content=msg)]})
        reply = extract_text(result["messages"][-1].content)
        trace = collect_trace(result["messages"])
        outputs = state.get("agent_outputs", []) + [("ELIGIBILITY_AGENT", reply, trace)]
        return {**state, "agent_outputs": outputs, "_next_agent_message": reply}

    def synthesise_node(state: GraphState) -> GraphState:
        """Supervisor synthesises all agent outputs into a final answer."""
        outputs = state.get("agent_outputs", [])
        if len(outputs) == 1:
            # Single agent — no need to re-summarise
            final = outputs[0][1]
        else:
            # Multiple agents — ask supervisor to synthesise
            context = "\n\n".join(
                f"[{name} output]:\n{reply}" for name, reply, _ in outputs
            )
            synth_msg = (
                f"Original user question: {state['user_message']}\n\n"
                f"Outputs from specialist agents:\n{context}\n\n"
                "Write a single clear, coherent response to the user that combines "
                "all the above information. Do not mention internal agent names."
            )
            response = supervisor_llm.invoke([HumanMessage(content=synth_msg)])
            final = extract_text(response.content)

        return {**state, "final_answer": final}

    # ── Route decision after supervisor ───────────────────────────────────────
    def route_after_supervisor(state: GraphState) -> str:
        agents = state.get("agents_to_call", [])
        if not agents:
            return "synthesise"
        first = agents[0]
        if first == "NUDGE_AGENT":       return "nudge"
        if first == "ELIGIBILITY_AGENT": return "eligibility"
        return "onboarding"

    def route_after_first_agent(state: GraphState) -> str:
        """Check if supervisor requested a second agent call."""
        agents_to_call = state.get("agents_to_call", [])
        outputs = state.get("agent_outputs", [])
        called_count = len(outputs)

        if called_count < len(agents_to_call):
            next_agent = agents_to_call[called_count]
            if next_agent == "ONBOARDING_AGENT": return "onboarding"
            if next_agent == "NUDGE_AGENT":       return "nudge"
            if next_agent == "ELIGIBILITY_AGENT": return "eligibility"
        return "synthesise"

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(GraphState)
    graph.add_node("supervisor",   supervisor_node)
    graph.add_node("onboarding",   run_onboarding_node)
    graph.add_node("nudge",        run_nudge_node)
    graph.add_node("eligibility",  run_eligibility_node)
    graph.add_node("synthesise",   synthesise_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "onboarding":  "onboarding",
        "nudge":       "nudge",
        "eligibility": "eligibility",
        "synthesise":  "synthesise",
    })

    # After any agent, check if another agent is needed or go to synthesis
    for agent_node in ["onboarding", "nudge", "eligibility"]:
        graph.add_conditional_edges(agent_node, route_after_first_agent, {
            "onboarding": "onboarding",
            "nudge":      "nudge",
            "eligibility":"eligibility",
            "synthesise": "synthesise",
        })

    graph.add_edge("synthesise", END)

    return graph.compile()

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🚗 Lyft Driver Onboarding")
st.caption("Multi-Agent AI System · Supervisor + 3 Specialists · LangGraph + Gemini")
st.divider()

pipeline = build_system()

tab1, tab2 = st.tabs(["🤖 Multi-Agent Chat", "ℹ️ Architecture"])

DRIVER_CHOICES = {
    f"{row['driver_id']} — {row['name']} ({row['city']}, {row['application_stage']})": row["driver_id"]
    for row in drivers_raw
}

BADGE = {
    "ONBOARDING_AGENT":  ("<span class='badge badge-onboarding'>ONBOARDING AGENT</span>", "🔵"),
    "NUDGE_AGENT":       ("<span class='badge badge-nudge'>NUDGE AGENT</span>",            "🟣"),
    "ELIGIBILITY_AGENT": ("<span class='badge badge-eligibility'>ELIGIBILITY AGENT</span>","🟢"),
    "SUPERVISOR":        ("<span class='badge badge-supervisor'>SUPERVISOR</span>",         "⬛"),
}

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        "The **Supervisor** reads your message, decides which specialist agent(s) to call "
        "(and in what order), then synthesises their outputs into a single answer."
    )

    col1, col2 = st.columns([3, 1])
    selected_label = col1.selectbox(
        "Driver context (optional — leave for pre-signup questions)",
        ["None — prospective / general"] + list(DRIVER_CHOICES.keys())
    )
    driver_id = "" if selected_label.startswith("None") else DRIVER_CHOICES[selected_label]

    st.markdown("**Quick scenarios to try:**")
    c1, c2, c3 = st.columns(3)
    scenarios = [
        ("📄 Doc status",          "What's still missing from my application?"),
        ("🔍 Full status check",   "Where do I stand overall — docs, background, everything?"),
        ("📣 Run nudge check",     "Run today's stalled driver nudge check."),
        ("🚗 Car eligibility",     "I have a 2010 Honda Civic in Chicago. Can I drive for Lyft?"),
        ("🔗 Chained: nudge+esc",  "Find stalled drivers and escalate any that are flagged."),
        ("📋 Policy question",     "What documents do I need and how long does the background check take?"),
    ]
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""
    for i, (label, msg) in enumerate(scenarios):
        col = [c1, c2, c3][i % 3]
        if col.button(label, key=f"sc_{i}", use_container_width=True):
            st.session_state.prefill = msg

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"], unsafe_allow_html=True)
            if turn.get("detail"):
                with st.expander("🔍 Agent trace"):
                    st.markdown(f"<div class='trace-box'>{turn['detail']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything — eligibility, onboarding status, nudge check…")
    if not user_input and st.session_state.prefill:
        user_input = st.session_state.prefill
        st.session_state.prefill = ""

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Supervisor routing…"):
                try:
                    result = pipeline.invoke({
                        "user_message": user_input,
                        "driver_id": driver_id,
                        "supervisor_reasoning": "",
                        "agents_called": [],
                        "agent_outputs": [],
                        "final_answer": "",
                    })

                    # Build trace for expander
                    trace_lines = []
                    trace_lines.append(f"SUPERVISOR REASONING:\n{result.get('supervisor_reasoning','')}\n")
                    trace_lines.append(f"AGENTS CALLED: {' → '.join(result.get('agents_to_call',[]))}\n")
                    for agent_name, reply, agent_trace in result.get("agent_outputs", []):
                        badge_html, emoji = BADGE.get(agent_name, ("", ""))
                        trace_lines.append(f"\n{'─'*50}")
                        trace_lines.append(f"{emoji} {agent_name} TOOLS:\n{agent_trace}")

                    # Build display message with agent badges
                    outputs = result.get("agent_outputs", [])
                    badge_row = " ".join(BADGE[a[0]][0] for a in outputs)
                    final = result.get("final_answer", "No response generated.")

                    display = f"{badge_row}\n\n{final}" if badge_row else final
                    st.markdown(display, unsafe_allow_html=True)

                    full_trace = "\n".join(trace_lines)
                    with st.expander("🔍 Supervisor + agent trace"):
                        st.markdown(f"<div class='trace-box'>{full_trace}</div>", unsafe_allow_html=True)

                    st.session_state.history.append({
                        "role": "assistant",
                        "content": display,
                        "detail": full_trace,
                    })

                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.history.append({"role": "assistant", "content": err})

    if st.session_state.history and st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.rerun()

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
## Architecture

```
User message
      │
      ▼
┌─────────────────────────────────────┐
│         SUPERVISOR AGENT            │
│  - reads message                    │
│  - reasons about routing            │
│  - decides which agents to call     │
│  - can chain agents sequentially    │
│  - synthesises final answer         │
└──────┬──────────┬───────────┬───────┘
       │          │           │
       ▼          ▼           ▼
 ┌──────────┐ ┌────────┐ ┌───────────┐
 │ONBOARDING│ │ NUDGE  │ │ELIGIBILITY│
 │  AGENT   │ │ AGENT  │ │   AGENT   │
 │          │ │        │ │           │
 │ 5 tools  │ │2 tools │ │ 2 tools   │
 └──────────┘ └────────┘ └───────────┘
       │          │           │
       └──────────┴───────────┘
                  │
                  ▼
         Synthesised answer
              to user
```

## What makes it truly multi-agent

- **Supervisor has its own LLM reasoning** — it doesn't use hardcoded if/else routing
- **Agents can be chained** — supervisor can call Agent A, pass its output to Agent B
- **Cross-agent workflows** — e.g. *"find stalled drivers and escalate any flagged ones"* triggers Nudge Agent → Onboarding Agent in sequence
- **Single entry point** — everything flows through one chat interface; the user never needs to know which agent handled what

## Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Routes and synthesises | — |
| **Onboarding Agent** | Existing driver support | `check_documents`, `run_background_check`, `schedule_inspection`, `lookup_policy`, `create_escalation_ticket` |
| **Nudge Agent** | Proactive stall detection | `get_stalled_drivers`, `send_nudge`, `create_escalation_ticket` |
| **Eligibility Agent** | Pre-signup vehicle check | `check_vehicle_eligibility`, `lookup_policy` |

## Stack
- **LLM:** Gemini 1.5 Flash
- **Agent framework:** LangGraph `create_react_agent` + `StateGraph`
- **Supervisor pattern:** LLM-driven routing with conditional edges
- **UI:** Streamlit
""")
