# Engineering Workflow: Agent + Reviewer

> **適用專案**：Cliff Walking — Q-learning vs SARSA 強化學習比較實驗  
> **版本**：2.0 ｜ **更新**：2026-04-25

---

## 📋 概述

本專案採用 **AI Agent（代理）+ Human Reviewer（人工審核者）** 協作開發流程。

- **Agent**：負責根據需求自動生成程式碼、文件與圖表
- **Reviewer**：負責審查邏輯正確性、學術品質與作業要求符合度

```
需求 → Agent 開發 → Reviewer 審查 → 修改 → 驗收
         ↑___________________________|（迭代）
```

---

## 🤖 Agent 職責

### 1. 程式碼生成與實作

Agent 根據作業規格自動實作功能，並遵守以下規範：

- ✅ 演算法公式正確（Q-learning / SARSA 更新式）
- ✅ 遵循 PEP 8 編碼風格
- ✅ 函式必須附帶完整 Docstring
- ✅ 關鍵邏輯加上行內注解
- ✅ 保留邊界條件處理（格子邊界、懸崖偵測）

**範例：合格的函式規格**

```python
def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-learning 演算法實作（Off-policy）。

    更新公式：Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]

    Args:
        env: CliffWalkingEnv 環境物件
        episodes (int): 訓練回合數，預設 500
        alpha (float): 學習率 α，預設 0.1
        gamma (float): 折扣因子 γ，預設 0.9
        epsilon (float): ε-greedy 探索率，預設 0.1

    Returns:
        Q (np.ndarray): 最終狀態-動作價值表，shape = (n_states, n_actions)
        rewards (list[float]): 每回合累積獎勵
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        total_r = 0
        done = False
        while not done:
            a = epsilon_greedy(Q, s, epsilon)
            s2, r, done, _ = env.step(a)
            # Off-policy: 使用最大 Q 值更新，不依賴實際執行的動作
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s = s2
            total_r += r
        rewards.append(total_r)
    return Q, rewards
```

### 2. 圖表生成

Agent 負責生成符合 Sutton & Barto 風格的圖表：

| 圖表 | 規格 |
|------|------|
| `reward_compare.png` | 含原始曲線（實線）與移動平均（虛線） |
| `reward_with_ci.png` | 含 ±1σ 陰影的信賴區間版 |
| `policy_q.png` | Q-learning 策略圖，含最優軌跡 |
| `policy_sarsa.png` | SARSA 策略圖，含最優軌跡 |

### 3. 文件生成

- `README.md`：完整的繁體中文專案說明
- `產品規格書.md`：對應作業需求的功能規格
- `ENGINEERING_WORKFLOW.md`：本文件

### 4. 版本管理

```bash
git commit -m "feat: implement Q-learning and SARSA with Cliff Walking env"
git commit -m "fix: correct SARSA on-policy action selection"
git commit -m "docs: update README with experiment results"
```

---

## 👁️ Reviewer 職責

### 1. 演算法正確性審查

**必查清單**：

```
☐ Q-learning 使用 max Q(s') 而非 Q(s', a')
☐ SARSA 的 a' 確實由 ε-greedy 從 s' 選出，非貪婪最大值
☐ ε-greedy 實作中隨機與貪婪的比例正確
☐ 獎勵機制：正常步 = -1，懸崖 = -100，到達 Goal = -1 + done
☐ 格子邊界：嘗試出界時代理停在原地，非循環繞回
☐ 懸崖偵測：掉入懸崖後代理確實回到 Start（而非 Goal 附近）
```

**常見錯誤範例**：

```python
# ❌ 錯誤：SARSA 卻用 max 來更新（這是 Q-learning）
Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])

# ✅ 正確：SARSA 使用實際下一步動作 a2
a2 = epsilon_greedy(Q, s2, epsilon)
Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
```

```python
# ❌ 錯誤：Q 表在 episode 內部重置
for ep in range(episodes):
    Q = np.zeros(...)  # 永遠學不了任何東西

# ✅ 正確：Q 表只在訓練開始前初始化一次
Q = np.zeros((env.n_states, env.n_actions))
for ep in range(episodes):
    ...
```

### 2. 圖表品質審查

| 審查項目 | 標準 |
|---------|------|
| 標題格式 | 必須包含 Epsilon、Alpha、run 次數 |
| 曲線顏色 | SARSA=青色（cyan），Q-learning=紅色（red） |
| Y 軸方向 | 負值在下，0 在上（獎勵為負） |
| 虛線含義 | 移動平均（平滑化），非另一個演算法 |
| 圖例完整 | 至少包含 Sarsa / Q-learning / 對應虛線說明 |

### 3. 作業需求符合度審查

```
☐ 使用 4×12 網格（或說明原因）
☐ 至少訓練 500 回合
☐ 有繪製每回合累積獎勵曲線
☐ 有策略視覺化
☐ 有分析收斂速度差異
☐ 有分析穩定性差異
☐ 有說明 Off-policy vs On-policy 理論差異
☐ 有結論說明哪種演算法收斂較快、較穩定
```

### 4. 代碼品質審查

```
☐ 所有 public 函式有 docstring
☐ 沒有未使用的 import
☐ 沒有硬編碼的路徑（使用 os.path.join）
☐ 沒有多餘的 print debug 語句
☐ 檔案命名清晰（train_final.py，非 train_v2_new_fixed.py）
```

---

## 📊 工作流程詳解

### Phase 1：需求分析（Agent）

```
輸入：作業說明 PDF + 參考圖表（Sutton & Barto）
  ↓
Agent：解析需求，建立產品規格書
  ↓
Reviewer：確認所有需求都已涵蓋
  ↓
輸出：產品規格書.md ✅
```

**Reviewer 檢查清單**：
- [ ] 作業要求全數列入規格書
- [ ] 參數設定符合作業建議
- [ ] 輸出圖表規格與參考圖一致

### Phase 2：核心實作（Agent）

```
Agent：
  1. 實作 CliffWalkingEnv（cliff_env.py）
  2. 實作 Q-learning 與 SARSA（rl_algorithms.py）
  3. 初步測試邊界條件
  4. 提交 Code Review
  ↓
Reviewer：
  1. 驗證演算法更新公式
  2. 測試懸崖偵測邏輯
  3. 確認邊界條件處理
  4. 批准或請求修改
```

### Phase 3：訓練腳本與視覺化（Agent）

```
Agent：
  1. 實作多次重複訓練（runs）
  2. 計算平均獎勵與標準差
  3. 生成 reward_compare.png（仿 Sutton 圖）
  4. 生成策略視覺化圖
  ↓
Reviewer：
  1. 比對圖表與參考圖
  2. 確認 SARSA 曲線高於 Q-learning
  3. 確認策略圖顯示正確路徑差異
```

### Phase 4：文件完成（Agent）

```
Agent：
  1. 撰寫 README.md
  2. 完成產品規格書.md
  3. 更新工程流程文件
  ↓
Reviewer：
  1. 確認文件完整性
  2. 結論是否符合理論
  3. 最終驗收
```

---

## 🔄 Reviewer 反饋模板

```markdown
## 代碼審查反饋

**檔案**：rl_algorithms.py  
**審查日期**：2026-04-25  
**審查者**：[Reviewer 姓名]

### ✅ 通過項目
- [x] Q-learning 更新公式正確
- [x] SARSA 使用實際執行動作
- [x] ε-greedy 實作正確

### ⚠️ 需要修改
1. **問題**：SARSA 函式缺少 docstring
   **建議**：
   ```python
   def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
       """SARSA 演算法（On-policy）。更新公式：Q(s,a) ← ..."""
   ```

2. **問題**：policy_sarsa.png 路徑應偏上方，但目前與 Q-learning 相同
   **建議**：檢查 SARSA Q 表是否正確訓練足夠回合數

### 結論
- [ ] 請修改後重新提交
```

---

## ✅ 本專案執行紀錄

### 已完成的審查

| Phase | 內容 | Agent | Reviewer | 狀態 |
|-------|------|-------|----------|------|
| 1 | 環境實作（cliff_env.py） | ✅ | ✅ | 通過 |
| 2 | Q-learning 實作 | ✅ | ✅ | 通過 |
| 3 | SARSA 實作 | ✅ | ✅ | 通過 |
| 4 | 訓練腳本（train_final.py） | ✅ | ✅ | 通過 |
| 5 | 圖表生成（仿 Sutton 風格） | ✅ | ✅ | 通過 |
| 6 | Web 應用（GitHub Pages 靜態網頁） | ✅ | ✅ | 通過 |
| 7 | 文件整理（README, 規格書） | ✅ | ✅ | 通過 |

### 關鍵設計決策紀錄

| 決策 | 選擇 | 理由 |
|------|------|------|
| α 值 | 0.5（非作業建議的 0.1） | 與 Sutton & Barto 參考圖對齊 |
| runs 次數 | 50（預設） | 足夠平滑，與參考圖一致 |
| 移動平均窗口 | w=50 | 對應參考圖的虛線平滑程度 |
| 主訓練腳本 | train_final.py | 整合所有功能的最終版本 |

---

## 📌 品質指標

### 演算法正確性驗證

```
預期結果：
  SARSA 最終平均獎勵  ≈ −20 ~ −25（高於 Q-learning）
  Q-learning 最終平均獎勵 ≈ −35 ~ −45
  SARSA 策略：上方繞道路徑（遠離懸崖）
  Q-learning 策略：貼崖捷徑（底部第 2 行）
```

### 效能基準

```
500 episodes × 50 runs：
  目標：< 60 秒
  記憶體：< 200 MB
```

### 文件完整性

```
✅ README.md — 繁體中文完整說明
✅ 產品規格書.md — 功能需求對應表
✅ ENGINEERING_WORKFLOW.md — 本文件
```

---

## 🚀 部署檢查清單

在提交作業前，請確認：

```
□ 所有測試通過（手動驗證圖表方向與顏色）
□ train_final.py 可直接執行並產生 4 張圖
□ reward_compare.png 中 SARSA 線高於 Q-learning 線
□ policy 圖中可清楚看到兩種演算法的路徑差異
□ 無多餘版本檔案（train_v2.py 等已刪除）
□ requirements.txt 包含所有依賴
□ README.md 包含執行指令
□ 產品規格書.md 對應所有作業要求
```

---

**版本**：2.0 ｜ **最後更新**：2026-04-25 ｜ **維護者**：Project Team
