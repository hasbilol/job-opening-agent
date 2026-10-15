"""Graph definition: START -> planner -> searcher -> matcher -> mailer -> END."""

from __future__ import annotations

import logging
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import Settings
from .nodes import matcher, mailer, planner, searcher
from .state import JobHunterState

logger = logging.getLogger(__name__)

NODE_ORDER = ("planner", "searcher", "matcher", "mailer")


def build_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Compile the linear job-hunting pipeline.

    ``settings`` is optional: when omitted, each node resolves the environment
    lazily at invocation time, which keeps this module import-safe in CI.
    """
    graph = StateGraph(JobHunterState)

    graph.add_node("planner", partial(planner, settings=settings))
    graph.add_node("searcher", partial(searcher, settings=settings))
    graph.add_node("matcher", partial(matcher, settings=settings))
    graph.add_node("mailer", partial(mailer, settings=settings))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "matcher")
    graph.add_edge("matcher", "mailer")
    graph.add_edge("mailer", END)

    compiled = graph.compile()
    logger.debug("Compiled graph: %s", " -> ".join(("START", *NODE_ORDER, "END")))
    return compiled
