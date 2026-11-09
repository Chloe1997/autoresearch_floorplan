# 題目：晶片 floorplan（簡化版）

在固定大小的晶片面積（outline，W × H）上擺放 N 個電路模組（block，長方形，寬高固定）。
模組之間有連線（net），每條 net 連接兩個模組並帶有權重。目標：讓總線長最短。

## 規則
- 模組不能重疊、不能超出 outline。
- 你的程式輸出會先經過官方 legalizer（去重疊工具）：已合法的輸出不會被改動；不合法的會被推成合法，但通常會變差。
- 總線長 = Σ 權重 × 兩模組中心的曼哈頓距離（|Δx| + |Δy|）。
- 分數 = baseline 的總線長 ÷ 你的總線長（baseline = 1.00，越高越好）；失敗、逾時記 0.1；整體取幾何平均。

## 你要交的程式
一個 Python 檔，定義：

```python
def place(inst: dict, time_limit: float) -> list[list[float]]:
    # 回傳每個模組左下角 [x, y]
```

inst 格式：
```json
{"outline": [W, H],
 "blocks": [{"w": 12, "h": 8}, ...],
 "nets":   [{"a": 0, "b": 1, "w": 2}, ...],
 "variant": {"forbidden": [], "edge": [], "fixed": {}, "distance": "manhattan", "rotate": false}}
```
`variant` 欄位在練習版都是預設值；正式比賽開賽時才會公布啟用哪一條。

## 限制
- 每題 time_limit 秒（預設 20）、2 個 CPU thread。
- 只能用 Python 標準函式庫、NumPy、PyTorch（CPU）。不能連網。

## 提示
baseline 用 PyTorch 把「線長 + 重疊懲罰 + 出界懲罰」當 loss，Adam 跑 300 步（像彈簧：net 把模組拉近，重疊把模組推開）。
它刻意沒調好：常數懲罰權重、隨機初始化、固定步數、沒有任何後處理。
