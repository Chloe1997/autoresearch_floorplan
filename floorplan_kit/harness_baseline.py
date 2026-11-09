"""Baseline auto-research harness (pydantic-ai). Teams are expected to beat THIS.

Loop: write place.py -> evaluate on dev -> read feedback -> revise. Nothing else
(no memory compression, no experiment tree, no reflection, no parallel runs).

  export LLM_BASE_URL=http://<vllm-host>:8000/v1  LLM_MODEL=<model-name>  [LLM_API_KEY=...]
  python harness_baseline.py
Budget knobs (env): MAX_EVALS=30  MAX_REQUESTS=120  SOLVER_TIME_LIMIT=20  EVAL_JOBS=4  TOOL_RETRIES=5
"""
import json, os, shutil, sys, time, traceback
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext, UsageLimits, capture_run_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "fp"))
from runner import score  # noqa: E402

MAX_EVALS = int(os.getenv("MAX_EVALS", 30))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 120))
TIME_LIMIT = float(os.getenv("SOLVER_TIME_LIMIT", 20))
JOBS = int(os.getenv("EVAL_JOBS", 4))
TOOL_RETRIES = int(os.getenv("TOOL_RETRIES", 5))
WORK = ROOT / "work"
HISTORY = WORK / "history.jsonl"   # one line per eval; read by tools/plot_progress.py
RUN_ID = time.strftime("%Y%m%d-%H%M%S")

INSTRUCTIONS = f"""You improve a Python program `place.py` for a chip floorplan task.
First call read_task and read_current_best. Then repeat: try_place_py(code=<the complete file>)
-> look at the scores and messages -> improve. try_place_py saves AND evaluates the code in one step.
Higher score is better; the baseline scores 1.00.
Change one idea at a time so you know what helped. You have {MAX_EVALS} evaluations.
When the evaluation budget is used up, reply with a short summary of what worked."""


@dataclass
class State:
    n_evals: int = 0
    best: float = 0.0
    best_version: str = "baseline"
    log: list = field(default_factory=list)


def make_model():
    return OpenAIChatModel(os.getenv("LLM_MODEL", "littlelearner"),
                           provider=OpenAIProvider(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                                                   api_key=os.getenv("LLM_API_KEY", "EMPTY")))


def make_agent(model=None) -> Agent[State, str]:
    agent = Agent(model or make_model(), deps_type=State, instructions=INSTRUCTIONS, retries=TOOL_RETRIES)

    @agent.tool_plain
    def read_task() -> str:
        """Problem statement, rules, input/output format."""
        return (ROOT / "task.md").read_text(encoding="utf-8")

    @agent.tool
    def read_current_best(ctx: RunContext[State]) -> str:
        """Source code of the best place.py so far (initially the baseline)."""
        return (WORK / "best_place.py").read_text(encoding="utf-8")

    @agent.tool
    def try_place_py(ctx: RunContext[State], code: str = "", **extra) -> str:
        """Save a complete place.py and evaluate it on all dev instances.
        `code` is the full Python source. Returns overall score + per-instance details."""
        st = ctx.deps
        # small models invent extra args (time_limit=...) or rename `code`: be lenient instead of burning retries
        if not code:
            code = next((str(extra[k]) for k in ("file", "content", "source", "place_py", "program") if extra.get(k)), "")
        if not code.strip():
            return "No code received. Call try_place_py with code=<the complete place.py source>. Nothing was evaluated."
        ignored = sorted(k for k in extra if k not in ("file", "content", "source", "place_py", "program"))
        note = f"(note: ignored extra arguments {ignored}; the time limit is fixed by the evaluator)\n" if ignored else ""
        if st.n_evals >= MAX_EVALS:
            return "EVALUATION BUDGET EXHAUSTED. Stop and summarize."
        v = f"v{len(list(WORK.glob('v*.py'))):03d}"
        p = WORK / f"{v}.py"
        p.write_text(code, encoding="utf-8")
        st.n_evals += 1
        try:
            r = score(str(p), "dev", TIME_LIMIT, ROOT / "instances", JOBS)
        except Exception:                        # evaluator problem: tell the model, don't kill the run
            return f"{v}: evaluation error (counts as an eval {st.n_evals}/{MAX_EVALS}):\n" + traceback.format_exc()[-1500:]
        if r["score"] > st.best:
            st.best, st.best_version = r["score"], v
            shutil.copy(p, WORK / "best_place.py")
        st.log.append({"version": v, "score": round(r["score"], 4)})
        rec = {"run": RUN_ID, "t": time.time(), "eval": st.n_evals, "version": v,
               "score": round(r["score"], 4), "best": round(st.best, 4), "best_version": st.best_version,
               "per_instance": {x["inst"]: round(x["score"], 4) for x in r["per_instance"]}}
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        lines = [f'{x["inst"]}: score={x["score"]:.3f} sec={x["sec"]} legalizer_displacement={x["disp"]} {x["msg"][:300]}'
                 for x in r["per_instance"]]
        return note + (f"{v} SCORE={r['score']:.4f} (best {st.best:.4f} = {st.best_version}; "
                f"evals {st.n_evals}/{MAX_EVALS})\n" + "\n".join(lines))

    @agent.tool
    def history(ctx: RunContext[State]) -> str:
        """Scores of all evaluated versions."""
        return json.dumps(ctx.deps.log)

    return agent


def dump_messages(msgs, tail=8):
    """Save the whole conversation and print the last tool calls/returns so a bad call is visible."""
    (WORK / "messages.json").write_bytes(ModelMessagesTypeAdapter.dump_json(msgs, indent=1))
    parts = [p for m in msgs for p in getattr(m, "parts", [])
             if type(p).__name__ in ("ToolCallPart", "ToolReturnPart", "RetryPromptPart")]
    for p in parts[-tail:]:
        body = getattr(p, "args", None) if type(p).__name__ == "ToolCallPart" else getattr(p, "content", None)
        print(f"[{type(p).__name__}] {getattr(p, 'tool_name', '')} {str(body)[:300]}")
    print(f"full conversation -> {WORK / 'messages.json'}")


def main(model=None):
    if not (ROOT / "instances" / "ref.json").exists():
        sys.exit("instances/ref.json missing - run first:\n"
                 "  python fp/runner.py solvers/baseline_place.py --make-ref --jobs 4")
    WORK.mkdir(exist_ok=True)
    if not (WORK / "best_place.py").exists():
        shutil.copy(ROOT / "solvers" / "baseline_place.py", WORK / "best_place.py")
    st = State(); agent = make_agent(model)
    with capture_run_messages() as msgs:
        try:
            res = agent.run_sync("Start.", deps=st, usage_limits=UsageLimits(request_limit=MAX_REQUESTS))
            print(res.output)
        except Exception as e:                   # request budget exhausted / model error: keep best so far
            print("run stopped:", type(e).__name__, str(e)[:300])
            dump_messages(msgs)
    print(f"best dev score {st.best:.4f} ({st.best_version}) -> {WORK / 'best_place.py'}")
    print(json.dumps(st.log, indent=1))
    return st


if __name__ == "__main__":
    main()
