"""Run scripted conversations through the harness on the offline engine.

    python scripts/demo.py            # all scenarios
    python scripts/demo.py bonus      # one scenario
    python scripts/demo.py --trace    # also print each turn's understanding

No API token and no network are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.insurance_claims import Agent  # noqa: E402

SCENARIOS: dict[str, dict] = {
    "happy": {
        "title": "Demo test case: verify, remember the hint, resolve, process, email summary",
        "lines": [
            "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
            "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.",
            "What documents do I need and how do I send them?",
            "How long will it take after I submit them?",
            "No, that's all.",
            "Yes, please send it.",
        ],
    },
    "bonus": {
        "title": "Bonus: frustrated caller, de-escalation without bypassing verification",
        "lines": [
            "Hi, my name is Margaret Chen and I want to know why my claim was denied.",
            "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
            "Why do you even need all this?",
            "Fine. My date of birth is March 15, 1985 and the last four of my SSN are 4472.",
            "What do I do now?",
            "that's all, thanks",
            "skip it",
        ],
    },
    "refusal": {
        "title": "Bonus: knowing when to stop persuading and hand off",
        "lines": [
            "I'm not giving you any personal information. Just tell me about my claim.",
            "No. I refuse to verify anything.",
            "I said no.",
        ],
    },
    "scope": {
        "title": "Out-of-scope questions: decline, then offer a human, then transfer",
        "lines": [
            "What is RL?",
            "Come on, just explain reinforcement learning.",
            "Write me a poem about the ocean.",
            "What's the weather in Paris?",
        ],
    },
    "representative": {
        "title": "Representative caller: listing and policyholder consent gate",
        "lines": [
            "Hi, I'm David Chen, calling for my mother Margaret Chen about her denied claim.",
            "Her date of birth is 1985-03-15 and her phone number is 650-521-2836.",
            "OK, she just approved it.",
            "How much was the claim for?",
            "that's everything",
            "no thanks",
        ],
    },
    "memory": {
        "title": "Memory across phases: hint and email preference given before verification",
        "lines": [
            "Hi, it's about my auto claim. And please email me a summary when we're done.",
            "Margaret Chen",
            "margaret@email.com, and my date of birth is 03/15/1985",
            "thanks, that's all",
            "yes",
        ],
    },
    "ambiguous": {
        "title": "Ambiguous claim, then a second claim",
        "lines": [
            "Margaret Chen, born 1985-03-15, phone 650-521-2836",
            "I have a question about my healthcare claim",
            "the first one",
            "What about my dental claim, how much was paid?",
            "that's all",
            "no",
        ],
    },
}


def run(name: str, show_trace: bool) -> None:
    scenario = SCENARIOS[name]
    agent = Agent()
    session = agent.new_session()
    print("=" * 88)
    print(f"{name}: {scenario['title']}")
    print("=" * 88)
    print(f"AGENT  [{session.phase}] {session.history[0]['text']}\n")
    for line in scenario["lines"]:
        out = agent.handle(session, line)
        trace = out["trace"]
        print(f"CALLER {line}")
        print(f"AGENT  [{' -> '.join(trace['phase_path'])}] ({trace['directive']['act']}) {out['reply']}")
        if trace["memory_captured"]:
            print(f"       memory captured: {', '.join(trace['memory_captured'])}")
        if trace["memory_used"]:
            print(f"       memory used: {'; '.join(trace['memory_used'])}")
        tools = [f"{t['name']}{'' if t.get('allowed') else ' (DENIED)'}" for t in trace["tools"]]
        if tools:
            print(f"       tools: {', '.join(tools)}")
        if show_trace:
            print(json.dumps(trace["understanding"], indent=1))
        print()
    for message in session.outbox:
        print("--- email in outbox " + "-" * 60)
        print(f"To: {message['to_masked']}\nSubject: {message['subject']}\n\n{message['body']}\n")
    if session.handoff:
        print("--- handoff ticket " + "-" * 61)
        print(json.dumps(session.handoff, indent=1))
        print()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for scenario_name in args or list(SCENARIOS):
        run(scenario_name, "--trace" in sys.argv)
