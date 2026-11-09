"""Check the harness plumbing WITHOUT an LLM: a scripted fake model reads the task,
writes the random sanity solver, evaluates it once, then stops."""
import sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import harness_baseline as hb
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

CODE = (ROOT / "solvers" / "random_place.py").read_text(encoding="utf-8")


def scripted(messages, info: AgentInfo):
    rets = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
    if len(rets) == 0:
        return ModelResponse(parts=[ToolCallPart("read_task", {})])
    if len(rets) == 1:
        return ModelResponse(parts=[ToolCallPart("try_place_py", {"code": CODE})])
    return ModelResponse(parts=[TextPart("done: " + str(rets[-1].content).splitlines()[0])])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:          # never touch the real work/ folder
        hb.WORK = Path(td); hb.HISTORY = hb.WORK / "history.jsonl"
        st = hb.main(FunctionModel(scripted))
        assert st.n_evals == 1, st
        assert (hb.WORK / "v000.py").exists() and hb.HISTORY.exists()
    print("SMOKE TEST OK")
