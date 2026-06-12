# 研究定位與後續方向（對照 PoseGAM）

本文件整理 PoseGAM 出現後，本研究的重新定位、與其差異、建議主方向，以及具體的後續步驟。

## 一句話定位（建議）

> 在「**沒有 CAD 模型、只有少數參考 RGB 影像、backbone 凍結、場景含多物體且未乾淨分割**」的限制下，探討 VGGT 式 multi-view 架構能把 unseen 6D object pose 做到多好——主打**實用性與效率**，而非單純在 BOP 上刷 SOTA。

這個定位把目前被 PoseGAM 比下去的設計（RGB-only、4 視角、凍結、未分割）重新詮釋成**刻意的限制與貢獻**，而不是缺點。

## 與 PoseGAM 的差異對照

| 面向 | PoseGAM | 本研究（目前） | 對你的意義 |
|---|---|---|---|
| 幾何資訊 | 需 CAD mesh，渲染點圖＋點雲（PTv3），注入是性能關鍵 | RGB-only，無顯式幾何 | 差異最大的軸，可走 CAD-free |
| 參考視角數 | 10–20，FPS 取樣，越多越好 | 4 個固定視角 | 走 sparse-reference 效率角度 |
| backbone | 從 VGGT 權重 finetune（優於凍結） | 完全凍結，只訓 adapter＋head | 走參數高效適配角度 |
| query 輸入 | 單物體、前景已分割 | 多物體、含背景、未分割 | 走雜亂場景 / 免分割角度 |
| 物體表徵 | 無特別壓縮 | 每層壓成 32 個 object token | 緊湊表徵，可當效率賣點 |
| 結果 | BOP SOTA（AR 41.1），裸 VGGT 僅 16.3 | 待測 | 它是強 baseline，也是可借用的 trick 來源 |

## 建議主方向

**「CAD-free、少參考、凍結 backbone 的雜亂場景 unseen 6D pose」**，並把 PoseGAM 證明有效的「幾何注入」用 **CAD-free 的方式**重新實現（見下節核心點子）。

理由：
- 一次在 4 個軸上和 PoseGAM 區隔（無 CAD、少視角、凍結、未分割），novelty 不會被單篇蓋掉。
- 完全契合你現有的設計，不用打掉重練。
- 故事誠實：不是宣稱比它準，而是「在更弱的假設下能做到多少」，這對機器人/AR 落地更有意義。

## 核心新點子：CAD-free 的幾何注入

PoseGAM 最重要的發現是「注入幾何讓 VGGT 從 16.3 → 41.1」。但它的幾何來自 CAD。本研究若是 CAD-free，可以改用**基礎模型預測的幾何**當作幾何 token 來源：

- 用 VGGT 本身對 4 張參考圖預測的 depth / point map，或用單目 depth/normal 估計器，當作 pseudo-geometry。
- 仿照 PoseGAM 的關鍵 trick：把幾何特徵**重排成 view-map 格式**再經 conv 注入，而非直接相加（它證明直接加會崩）。
- 這直接把「幾何很重要」的洞見搬到 CAD-free 設定，是明確且可發表的貢獻點。

## 後續具體步驟（路線圖）

### 第 0 步：先決定一個分岔
確認你的應用場景**到底有沒有 CAD 模型**。
- 有 → 仍建議走 CAD-free 為主線，但可加一組「有 CAD」的對照實驗，借 PoseGAM 的幾何注入當上界參考。
- 沒有 → 直接主打 CAD-free，pseudo-geometry 注入成為核心貢獻。

### 第 1 步：把現有模型跑通並量測
在 BOP 子集（建議先 TUD-L、YCB-V，單物體較乾淨）上量出目前 RGB-only＋凍結＋4 視角的 AR / AUC，知道起點在哪。

### 第 2 步：支撐故事的消融
照 PoseGAM 的表格做你自己的版本，重點是證明你的**限制下的權衡**：
- 視角數：4 vs 8 vs 12（驗證你在少視角的位置）。
- 凍結 vs finetune（量化省下多少可訓練參數、犧牲多少精度）。
- 有無 pseudo-geometry 注入（這是你的核心點，務必做）。
- 取樣策略：固定 4 視角 vs FPS（借 PoseGAM 的 trick 看能否免費提升）。

### 第 3 步：選擇性借用 PoseGAM 的 trick
- view-format 幾何注入（而非直接相加）。
- FPS 取樣參考視角。
- 若允許，少量 finetune backbone 的 LoRA / adapter 版本（凍結與全 finetune 之間的折衷）。

### 第 4 步：強化差異化設定
- 多物體 / 免分割：測在雜亂場景直接估目標物體 pose。
- 接 PoseGAM 列出的 open 問題：非剛體 / 關節物體、透明反光物體。

## 風險與誠實的取捨

- 依 PoseGAM 消融，RGB-only＋凍結＋少視角**精度上大機率不如它**。所以貢獻**不要定位在「刷贏 BOP」**，要定位在「假設更弱 / 更省 / 更貼近落地」的權衡曲線上。
- 報告時建議畫「精度 vs 需求（視角數、是否需 CAD、可訓練參數量）」的權衡圖，讓 reviewer 看到你的位置而非只比單一數字。
- 一定要把 PoseGAM 當 baseline 跑/引用，並清楚說明你刻意拿掉了哪些假設。

## 投稿定位建議

把貢獻寫成三點：
1. 一個 CAD-free、凍結 backbone、少參考的 unseen 6D pose 框架（強調更弱假設）。
2. CAD-free 的幾何注入（用基礎模型預測幾何＋view-format 注入），把「幾何有效」的洞見帶到無 CAD 設定。
3. 在雜亂多物體 / 免分割（與/或非剛體、透明物體）設定下的評估與分析。

## 待你回覆 / 待確認

- 應用場景是否有 CAD 模型？（決定主線）
- 是否接受少量 finetune（LoRA/adapter），還是堅持完全凍結？
- 目標 benchmark 與算力預算（決定先跑哪些資料集與消融規模）。
