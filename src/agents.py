import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
from src.rag import retrieve

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========== STATE ==========
class AgentState(TypedDict):
    question: str
    chunks: list[dict]
    validation_status: str  # "sufficient" | "insufficient" | "pending"
    answer: str
    retry_count: int
    low_confidence: bool


# ========== NODES ==========
def retriever_node(state: AgentState) -> AgentState:
    """Retrieve top-5 chunks from ChromaDB for the question."""
    print(f"[Retriever] Searching for: {state['question'][:60]}...")
    chunks = retrieve(state["question"], n_results=5)
    return {
        **state,
        "chunks": chunks,
        "validation_status": "pending"
    }


def validator_node(state: AgentState) -> AgentState:
    """Check if retrieved chunks contain sufficient evidence to answer."""
    print(f"[Validator] Checking {len(state['chunks'])} chunks...")

    context = "\n\n".join([c["text"] for c in state["chunks"]])

    prompt = f"""You are a strict medical research validator.

Question: {state["question"]}

Retrieved context:
{context}

Your task: Determine if the context contains DIRECT and SPECIFIC evidence 
to answer the question accurately.

Rules:
- SUFFICIENT: context directly addresses the question with specific findings
- INSUFFICIENT: context is only tangentially related, or discusses a related 
  topic without answering the specific question

Reply with ONLY one word: SUFFICIENT or INSUFFICIENT."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    verdict = response.choices[0].message.content.strip().upper()
    status = "sufficient" if "SUFFICIENT" in verdict else "insufficient"
    print(f"[Validator] Verdict: {status} (retry_count={state['retry_count']})")

    return {
        **state,
        "validation_status": status,
        "retry_count": state["retry_count"] + 1
    }


def synthesizer_node(state: AgentState) -> AgentState:
    print(f"[Synthesizer] Generating answer...")

    context = "\n\n".join([
        f"[PubMed {c['pubid']} - {c['label']}]\n{c['text']}"
        for c in state["chunks"]
    ])

    prompt = f"""You are a medical research assistant.
Answer the question using ONLY the provided context.
Cite PubMed IDs inline.
If the context does not contain sufficient evidence, 
start your answer with "INSUFFICIENT_EVIDENCE:" and explain why.

Context:
{context}

Question: {state["question"]}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    
    # Detect low confidence answers
    low_confidence = any(phrase in answer.lower() for phrase in [
        "no direct evidence",
        "cannot be determined", 
        "insufficient_evidence",
        "not enough information",
        "context does not"
    ])
    
    print(f"[Synthesizer] Done. Low confidence: {low_confidence}")

    return {
        **state,
        "answer": answer,
        "low_confidence": low_confidence
    }


# ========== ROUTING ==========
def route_after_validation(state: AgentState) -> str:
    """Decide next node based on validation result."""
    if state["validation_status"] == "sufficient":
        return "synthesizer"
    elif state["retry_count"] >= 2:
        # Max retries reached — synthesize anyway
        print("[Router] Max retries reached. Synthesizing with available chunks.")
        return "synthesizer"
    else:
        return "retriever"


# ========== GRAPH ==========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("validator", validator_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "synthesizer": "synthesizer",
            "retriever": "retriever"
        }
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    question = "Do mitochondria play a role in programmed cell death?"
    result = app.invoke({
        "question": question,
        "chunks": [],
        "validation_status": "pending",
        "answer": "",
        "retry_count": 0
    })

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for c in result["chunks"]:
        print(f"  - PubMed {c['pubid']} ({c['label']})")