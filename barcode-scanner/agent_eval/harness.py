"""Run one agent in one fresh environment and score it."""
from __future__ import annotations

import json
import os
import time

from .agents import Agent, materialise
from .environment import Environment
from .scorer import Score, score
from .tools import ToolExecutor


def run_one(agent: Agent, base_dir: str, max_turns: int = 30, **env_kwargs) -> tuple[Score, dict]:
    run_dir = os.path.join(base_dir, agent.name.replace(":", "_"))
    env = Environment(
        workdir=os.path.join(run_dir, "work"),
        hidden_dir=os.path.join(run_dir, "hidden"),
        **env_kwargs,
    ).build()
    executor = ToolExecutor(env)
    materialise(agent, executor)

    t0 = time.perf_counter()
    traj = agent.run(executor, max_turns)
    traj.agent_name = agent.name  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - t0

    s = score(env, traj)
    record = {"score": s.to_dict(), "trajectory": traj.to_dict(), "seconds": round(elapsed, 1)}
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(record, f, indent=2)
    return s, record
