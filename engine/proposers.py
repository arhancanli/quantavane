"""Proposers write new candidate programs. Every proposer charges a Ledger with a hard cap; a local model costs $0."""
import json
import re
import urllib.request


class BudgetExceeded(Exception):
    pass


class Ledger:
    def __init__(self, cap_usd):
        self.cap, self.spent = float(cap_usd), 0.0

    def reserve(self, usd):
        if self.spent + usd > self.cap + 1e-12:
            raise BudgetExceeded(f"cap ${self.cap:.2f} reached (spent ${self.spent:.4f})")

    def charge(self, usd):
        self.spent += usd


SYSTEM = ("You are a research mathematician and expert Python programmer improving a program that constructs a "
          "solution to an optimisation problem. You may use numpy and scipy. Return ONE complete Python program in a "
          "single ```python block that defines construct(). No network access; read only the input files the problem names. Keep the runtime under the "
          "stated limit. Think about why the current program falls short, then make one substantial, concrete "
          "improvement: better starting points, better search moves and schedules, smarter use of structure "
          "(symmetry, group orbits, known constructions), multiple restarts.")

CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def build_prompt(problem, parent, inspirations, failures):
    parts = [f"PROBLEM\n{problem.statement()}",
             f"Best known value: {problem.record.value} ({problem.record.holder}); {'higher' if problem.direction == 'max' else 'lower'} is better. Runtime limit: {problem.timeout}s on one CPU core.",
             f"CURRENT PROGRAM (score {parent['score']:.6f})\n```python\n{parent['program']}\n```"]
    for ins in inspirations:
        parts.append(f"ANOTHER PROGRAM FOR IDEAS (score {ins['score']:.6f})\n```python\n{ins['program']}\n```")
    if failures:
        parts.append("RECENT FAILED ATTEMPTS ON THIS LINE:\n" + "\n".join(f"- {f['note'][:300]}" for f in failures))
    parts.append("Write the improved program now.")
    return "\n\n".join(parts)


def extract_program(text):
    blocks = CODE.findall(text or "")
    blocks = [b for b in blocks if "def construct" in b]
    return blocks[-1].strip() + "\n" if blocks else None


class OllamaProposer:
    """A model running locally through Ollama. Free, so it charges $0, but it still goes through the ledger."""

    def __init__(self, model, ledger, url="http://127.0.0.1:11434", temperature=0.8, name=None, num_ctx=16384):
        self.model, self.ledger, self.url, self.temperature = model, ledger, url, temperature
        self.name = name or f"ollama:{model}"
        self.num_ctx = num_ctx  # prompts run ~6k tokens; the model's own 32k context held 14 GB of a 24 GB Mac

    def propose(self, prompt):
        self.ledger.reserve(0.0)
        body = json.dumps({"model": self.model, "stream": False, "think": False, "keep_alive": "5m",
                           "options": {"temperature": self.temperature, "num_predict": 4096, "num_ctx": self.num_ctx},
                           "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", body, {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            text = json.load(r)["message"]["content"]
        self.ledger.charge(0.0)
        return extract_program(text), 0.0
