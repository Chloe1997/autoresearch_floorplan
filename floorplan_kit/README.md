# Floorplan AutoResearch 黑克松 — baseline kit（學生簡化版）

```
fp/problem.py            測資生成、合法性檢查、線長計算、legalizer（去重疊工具）
fp/runner.py             在 subprocess 跑 place.py（含 timeout）→ legalizer → 計分
solvers/baseline_place.py 官方 baseline（PyTorch + Adam，刻意沒調好，分數定義為 1.00）
solvers/random_place.py   環境檢查用（隨機擺放，不需 PyTorch）
harness_baseline.py       pydantic-ai 的 auto-research harness baseline
task.md                   給 agent / 學生看的題目說明
tools/convert_iccad04.py  ICCAD04（IBM）benchmark → 本題 JSON
instances/dev, hidden     已產生好的合成測資（各 12 題，20–200 個模組）
```

## 1. 安裝（Python 3.10 以上）

```bash
pip install numpy torch "pydantic-ai-slim[openai]"
```

離線環境：先在能上網的機器 `pip download numpy torch "pydantic-ai-slim[openai]" -d wheels`，
把 `wheels/` 帶進去後 `pip install --no-index --find-links wheels numpy torch "pydantic-ai-slim[openai]"`。

## 2. 跑 baseline（不需要 LLM）

```bash
# (a) 用 baseline 算出每題的參考線長 → instances/ref.json
python fp/runner.py solvers/baseline_place.py --make-ref --jobs 4

# (b) 再跑一次 baseline，應該得到 SCORE[dev] = 1.0000（確認結果可重現）
python fp/runner.py solvers/baseline_place.py --jobs 4

# (c) 對照組：隨機擺放 + legalizer，分數應該低於 1
python fp/runner.py solvers/random_place.py --jobs 4

# (d) 主辦方：看 hidden 分數
python fp/runner.py solvers/baseline_place.py --split hidden --jobs 4
```

每行輸出：`score`（ref/你的線長）、`sec`（執行秒數）、`disp`（legalizer 把模組推了多遠，越小代表輸出越接近合法）。

## 3. 跑 harness（需要 LLM）

```bash
python tests/smoke_harness.py            # 先用假模型檢查流程，最後印出 SMOKE TEST OK

export LLM_BASE_URL=http://<vllm-host>:8000/v1   # 任何 OpenAI 相容端點（vLLM、Ollama…）
export LLM_MODEL=<model-name>
export LLM_API_KEY=EMPTY
export PYDANTIC_AI_NO_BANNER=1
MAX_EVALS=30 MAX_REQUESTS=120 python harness_baseline.py
```

產出在 `work/`：每個版本 `v000.py, v001.py…`，最佳版本 `work/best_place.py`。
用 `python fp/runner.py work/best_place.py --split hidden` 看它在 hidden 上的分數。

## 4.（可選）加入真實設計 ICCAD04

```bash
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026 && git submodule update --init external/MacroPlacement && cd ..
python tools/convert_iccad04.py macro-place-challenge-2026/external/MacroPlacement/Testcases/ICCAD04 \
       --all -o instances/dev_ibm --whitespace 0.15
python fp/runner.py solvers/baseline_place.py --make-ref     # 重新算 ref（含新資料夾）
python fp/runner.py solvers/baseline_place.py --split dev_ibm
```

hidden 用衍生題：加 `--perturb <seed>`。ibm 題目較大（178–786 個模組），注意 time limit。

## 產生新的合成測資

```bash
python fp/problem.py --out instances          # dev + hidden 各 12 題
python -c "import sys; sys.path.insert(0,'fp'); import problem, json; \
json.dump(problem.generate(300, 7, whitespace=0.10), open('instances/hidden/syn300_s7.json','w'))"
```

變體（禁區、貼邊、預放模組、距離定義、旋轉）都可以在 `generate(..., variant={...})` 打開；
checker 與 legalizer 都已支援。換測資後記得重跑 `--make-ref`。
