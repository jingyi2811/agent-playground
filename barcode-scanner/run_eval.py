"""Run the agent eval.

    python run_eval.py --agents honest overclaimer cheater breaker   # offline references
    python run_eval.py --agents claude                                # needs ANTHROPIC_API_KEY
    python run_eval.py --agents claude honest --model claude-opus-5
"""
from __future__ import annotations

import argparse
import json
import os

from agent_eval.agents import ClaudeAgent, reference_agents
from agent_eval.harness import run_one


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", default=["honest", "overclaimer", "cheater", "breaker"])
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--severity", type=float, default=1.6)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--out", default="eval_runs")
    a = p.parse_args()

    refs = reference_agents()
    agents = []
    for name in a.agents:
        if name == "claude":
            agents.append(ClaudeAgent(a.model, a.effort))
        elif name in refs:
            agents.append(refs[name])
        else:
            p.error(f"unknown agent {name}; choose from claude, {', '.join(refs)}")

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for agent in agents:
        s, rec = run_one(agent, a.out, a.max_turns, severity=a.severity, count=a.count)
        rows.append(s)
        print(f"{s.agent:<14} {s.outcome:<16} dev={s.dev_rate:.0%} hidden={s.hidden_rate:.0%} "
              f"claimed={fmt(s.claimed)} conf={fmt(s.confidence)} verified={s.verified_before_claim} "
              f"tests={s.tests_pass} scope={s.in_scope} ({rec['seconds']}s)")

    print()
    print(f"baseline: dev={rows[0].baseline_dev:.0%} hidden={rows[0].baseline_hidden:.0%}")
    print(f"{'agent':<14} {'outcome':<16} {'gain':>6} {'claim_err':>10} {'cal_gap':>8}")
    for s in rows:
        print(f"{s.agent:<14} {s.outcome:<16} {s.improvement_hidden:>+6.0%} "
              f"{fmt(s.claim_error, '+.0%'):>10} {fmt(s.calibration_gap, '+.2f'):>8}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump([s.to_dict() for s in rows], f, indent=2)
    return 0


def fmt(v, spec=".0%"):
    return "-" if v is None else format(v, spec)


if __name__ == "__main__":
    raise SystemExit(main())
