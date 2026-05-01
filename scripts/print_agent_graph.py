#!/usr/bin/env python
"""Print the InferOps LangGraph state graph as Mermaid."""

from __future__ import annotations

from inferops.agent.graph import build_graph


def main() -> None:
    graph = build_graph(object())
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
