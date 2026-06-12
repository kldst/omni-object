# CLAUDE.md

本檔案提供本專案的架構與設計慣例，供 Claude（含 Claude Code）理解程式碼脈絡時參考。

## 專案目標

以 VGGT 為 backbone，設計一個 **6D object pose estimation** 模型。給定單張 scene image 與目標物體的多視角參考圖，預測該物體相對於相機的 6D pose。

## 輸入

- **Scene image**：單張該視角影像，含背景與多個物體。
- **Object reference images**：對每個目標物體，在 Blender 中以虛擬相機從 4 個固定視角拍攝（上方、前方、偏下方、下方），共 4 張。

## 架構總覽

雙分支共用同一個 **frozen pretrained VGGT**（alternating-attention backbone）。可訓練模組只有壓縮網路、cross-attention 與 pose head。

```
Object branch:
  4 object images → [frozen VGGT] → 抽第 {4,11,17,23} 層 tokens
                  → [compress net] → 每層 32 個 object tokens（共 4 組）

Scene branch:
  1 scene image → [frozen VGGT，於 {4,11,17,23} 層注入 cross-attention]
                → 第 23 層輸出 fused tokens → [pose head] → 6D pose
```

### Object branch（編碼物體多視角先驗）

1. 4 張 object image 一起送入 frozen VGGT。
2. 從 AA backbone 的第 **{4, 11, 17, 23}** 層各抽出一組 tokens。
3. 每層 tokens 經壓縮網路降維為 **32 個 object tokens**，得到 4 組（每層一組）。

### Scene branch + 融合（將物體先驗注入場景特徵）

1. 單張 scene image 送入同一個 frozen VGGT。
2. scene tokens 行進到第 **{4, 11, 17, 23}** 層時，與該層對應的 32 個 object tokens 做 **cross-attention**，把物體資訊注入場景表徵。
3. 第 23 層輸出 **fused tokens**。

### Pose head

以 fused tokens decode 出物體的 6D pose。

## 關鍵慣例與常數

- **層索引**：`{4, 11, 17, 23}`，與 VGGT 原論文一致（附錄 B：feed 第 4/11/17/23 個 block 的 tokens 進 DPT）。抽取與注入皆使用同一組層。
- **object tokens 數量**：每層 32 個。
- **參考視角數**：4（上 / 前 / 偏下 / 下）。
- **Frozen vs trainable**：
  - Frozen：pretrained VGGT 全部權重。
  - Trainable：壓縮網路、cross-attention 模組、pose head。
  - 梯度只流經上述可訓練模組，不更新 VGGT 本體。

## 待確認 / TODO

- 壓縮網路是「4 層各自一個」還是「4 層共用一個」，需於程式碼與文件中明確。
- pose head 的具體輸出參數化（例如 quaternion + translation，或其他 6D 表示法）尚待定義。
- 4 個參考視角集中於單一半球，是否需補充視角以涵蓋更完整的物體外觀，待評估。
