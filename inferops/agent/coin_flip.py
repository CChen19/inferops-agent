"""
LangGraph warm-up: Coin Flip Agent.

Graph structure:
  START -> decide -> [heads: execute_heads | tails: execute_tails] -> decide (loop)
                                                                   -> END  (max flips reached)

Teaches:
  - TypedDict state shared across nodes
  - StateGraph / add_node / add_conditional_edges
  - loop with a termination condition
  - interrupt (human-in-the-loop pause before each flip)
"""

from __future__ import annotations

import random
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CoinFlipState(TypedDict):
    flip_number: int
    max_flips: int
    results: list[str]        # history of "heads" / "tails"
    last_outcome: str         # most recent flip
    heads_count: int
    tails_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def decide(state: CoinFlipState) -> CoinFlipState:
    """Flip the coin and record the outcome."""
    outcome = random.choice(["heads", "tails"])
    return {
        **state,
        "flip_number": state["flip_number"] + 1,
        "last_outcome": outcome,
        "results": state["results"] + [outcome],
        "heads_count": state["heads_count"] + (1 if outcome == "heads" else 0),
        "tails_count": state["tails_count"] + (1 if outcome == "tails" else 0),
    }


def execute_heads(state: CoinFlipState) -> CoinFlipState:
    """Action for a heads outcome."""
    print(f"  Flip {state['flip_number']}: HEADS  (H={state['heads_count']} T={state['tails_count']})")
    return state


def execute_tails(state: CoinFlipState) -> CoinFlipState:
    """Action for a tails outcome."""
    print(f"  Flip {state['flip_number']}: TAILS  (H={state['heads_count']} T={state['tails_count']})")
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_decide(state: CoinFlipState) -> Literal["execute_heads", "execute_tails"]:
    return "execute_heads" if state["last_outcome"] == "heads" else "execute_tails"


def route_after_execute(state: CoinFlipState) -> Literal["decide", "__end__"]:
    if state["flip_number"] >= state["max_flips"]:
        return END
    return "decide"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(interrupt_before_flip: bool = False) -> StateGraph:
    g = StateGraph(CoinFlipState)

    g.add_node("decide", decide)
    g.add_node("execute_heads", execute_heads)
    g.add_node("execute_tails", execute_tails)

    g.add_edge(START, "decide")

    # conditional edge: decide -> execute_heads or execute_tails
    g.add_conditional_edges("decide", route_after_decide)

    # conditional edge: execute_* -> decide (loop) or END
    g.add_conditional_edges("execute_heads", route_after_execute)
    g.add_conditional_edges("execute_tails", route_after_execute)

    if interrupt_before_flip:
        # Demonstrates human-in-the-loop: pause before each flip decision
        return g.compile(interrupt_before=["decide"])

    return g.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(max_flips: int = 5, interrupt: bool = False) -> CoinFlipState:
    graph = build_graph(interrupt_before_flip=interrupt)

    initial: CoinFlipState = {
        "flip_number": 0,
        "max_flips": max_flips,
        "results": [],
        "last_outcome": "",
        "heads_count": 0,
        "tails_count": 0,
    }

    print(f"=== Coin Flip Agent — {max_flips} flips ===")

    if interrupt:
        # Step through manually — simulates human approval of each flip
        config = {"configurable": {"thread_id": "demo"}}
        state = graph.invoke(initial, config)
        while state["flip_number"] < max_flips:
            print(f"  [interrupt] about to flip #{state['flip_number'] + 1} — resuming…")
            state = graph.invoke(None, config)
    else:
        state = graph.invoke(initial)

    print(f"\nFinal: {state['heads_count']} heads / {state['tails_count']} tails")
    print(f"Sequence: {' '.join(r[0].upper() for r in state['results'])}")
    return state


if __name__ == "__main__":
    run(max_flips=8)
