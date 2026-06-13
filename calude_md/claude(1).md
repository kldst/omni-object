# OmniVGGT 模型架構 — `train_hc_diverse24.py` (0612, no_pooler baseline)

> 本文件對應 config: [`configs/train_hc_diverse24.py`](../../configs/train_hc_diverse24.py)
> 實驗名稱: `hc_only_diverse24_14k_0612_no_pooler_revert`
> 任務: HouseCat6D camera-frame 6D 物件姿態估計
>
> **本實驗已還原回 `no_pooler` baseline**:Query Pooler + pose decoder query 替換經實驗
> 確認**沒有幫助**,已關閉(`enable_object_query_pooler=False`)。attn mask loss、
> zero-init gate 也維持關閉。換言之,目前的 pose head 用的是原本**單一 learnable query**
> 的設計,object 特徵只透過 4 層 cross-attn 注入 scene。
>
> 標記說明:
> - ✅ **本 config 啟用**
> - ⛔ **已實作但本 config 關閉**(flag = False,程式碼留在 repo 休眠,可隨時重新啟用)

---

## 1. 概觀

這是一個 **two-tower(雙編碼器)6D 物件姿態估計** 架構,基於 VGGT
(Visual Geometry Grounded Transformer)aggregator:

1. **DINOv2 ViT-L/14 backbone** 從 scene 影像與 object 參考影像抽 patch 特徵。
2. **VGGT aggregator**(24 層,frame/global 交替注意力)融合多視角 scene 資訊。
3. **多層 object cross-attention**(layer 4/11/17/23)把 object 特徵漸進式注入 scene。
4. **三個 head** 解碼最後一層 scene tokens:**mask / presence / pose(SRT)**。

cross-attention 是 scene→object 方向的「廣播式」注入,背景 tokens 也會拿到物件特徵。
曾規劃用 Query Pooler + object-conditioned pose query 讓 pose head 結構性聚焦在物體區域,
但實驗顯示沒有幫助,已全部關閉。本 config 各項改動的狀態:

1. ⛔ **Query pooler**(`enable_object_query_pooler=False`):已還原。pose head 退回原本的
   單一 learnable query,不再把 object tokens 壓成 32 個 object queries。
2. ⛔ **Pose decoder query 替換**:隨 Query pooler 一起關閉(flag off 時 `object_query_aggregation`
   不生效)。decoder 用 `TransformerDecoder(num_tokens=1, token_dim=1)` 的單一 query。
3. ⛔ **Attn mask loss**(`enable_attn_mask_loss=False`):未啟用。
4. ⛔ **Zero-init gate**(`object_cross_attn_zero_gate=False`):未啟用。
   warm-start 自 `checkpoint-1-688`,其 cross-attn 區塊已是訓練過的,α=0 反而會在 step 0
   把學到的注入貢獻歸零(與「保護 warm-start 特徵」的初衷相反),故刻意關閉。

> ⚠️ 本 config 把 **depth / point / camera** 三個 head 全部關閉
> (`enable_depth/point/camera = False`,對應 loss weight 也都是 0)。
> 並且 `disable_object_prototype_pooler = True`,所以 cross-attn 的 K/V **不經 pooler**
> (`no_pooler`)。Query pooler 的程式碼仍在 repo([object_pose_head.py](../../omnivggt/heads/object_pose_head.py)),
> 只是 flag 關閉、不會被建構,要重新對照時把 `enable_object_query_pooler=True` 即可。

---

## 2. 架構圖(本 config:no_pooler baseline,query pooler/attn loss/gate 全 ⛔)

```
                          ┌─────────────────────────────────────────────────────┐
   INPUT                  │  object_images  (B, S_obj=4, 3, H, W)                 │
   ─────                  │  4 個參考視角: top / front / back / bottom            │
                          │  (fixed_object_view_ids = 0,5,8,19)                   │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                          ┌─────────────────────────────────────────────────────┐
                          │   FROZEN Object Encoder  (凍結, bf16)                 │
                          │   = aggregator 的一份拷貝, requires_grad=False        │
                          └───────────────────────┬─────────────────────────────┘
                                                  │ 抽取 layer {4,11,17,23} 的
                                                  │ object patch tokens (no pooler)
                                                  ▼
                            object_prototypes_by_idx = {4:…, 11:…, 17:…, 23:…}
                                     │
                     (完整 tokens 當 cross-attn K/V)
                                     │
   ┌──────────────────────┐          │            ┌──────────────────────────┐
   │ scene images         │          │            │ ⛔ Query Pooler (關閉)     │
   │ (B, S=50, 3,518,476) │          │            │   flag off,不建構;       │
   └──────────┬───────────┘          │            │   pose head 用單一 query  │
              ▼                      │            └──────────────────────────┘
   ┌──────────────────────────┐      │
   │ DINOv2 PatchEmbed (凍結)  │      │
   └──────────┬───────────────┘      │
              ▼                      │
   ╔══════════════════════════════════════════╗
   ║   VGGT Aggregator  (24 層, 可訓練)         ║
   ║   Frame-Attn / Global-Attn 交替            ║
   ║   ┌─ layer 4  ─► [Obj Cross-Attn]        ◄╫─ Q=scene patch
   ║   ├─ layer 11 ─► [Obj Cross-Attn]        ◄╫─ K/V=obj tokens
   ║   ├─ layer 17 ─► [Obj Cross-Attn]        ◄╫─ 16 heads
   ║   └─ layer 23 ─► [Obj Cross-Attn]        ◄╫─ (gate ⛔ 關閉)
   ╚══════════════════════╤═════════════════════╝
                          │ 最後一層 scene tokens (B, S, P, C)
        ┌─────────────────┤
        ▼                 ▼ (flatten 成 K/V)
 ┌─────────────────┐   ┌──────────────────────────────────────┐
 │ Object Mask Head │   │ Object Pose Decoder (baseline)        │
 │ Conv C→256→128→1 │   │ TransformerDecoder depth=6, heads=8   │
 │ → object_mask    │   │ Q = 1 learnable query (num_tokens=1)  │
 │   (B,S,H,W)      │   │ K/V = flatten scene tokens            │
 └─────────────────┘   │ + IEF ×1                              │
                        │   │  (⛔ attn_mask_loss 未啟用)         │
                        │   ▼                                   │
                        │   ├─ decpose      → rot6d(6)           │
                        │   ├─ dectranslate → trans(3)           │
                        │   ├─ decsize      → size_log(3)        │
                        │   └─ presence_branch → logit           │
                        └──────────────────────────────────────┘

 LOSS (全部加權相加):
   object_mask_loss      = BCE + Dice                  (weight 1.0)
   object_presence_loss  = BCE                         (weight 1.0)
   object_srt_loss       = L1(rot6d + trans + size), symmetric_rot6d
   ⛔ attn_mask_loss      = 未啟用 (enable_attn_mask_loss=False)
   ⛔ relative_pose_loss  = 未啟用 (enable_relative_pose_loss=False)
```

---

## 3. 元件對照表

| 元件 | 檔案 | 行數 | 狀態 |
|------|------|------|------|
| 主模型 `OmniVGGT` | `omnivggt/models/omnivggt.py` | 79–565 | 既有 |
| 建構子 | `omnivggt/models/omnivggt.py` | 80–207 | 既有 |
| 模型載入 `load_model` | `train_utils.py` | 229–350 | 既有 |
| Aggregator (VGGT) | `omnivggt/models/aggregator.py` | 25–462 | 既有 |
| `ZeroAggregator` | `omnivggt/models/omnivggt_aggregator.py` | 19–450 | 既有 |
| Layer postprocessor 注入點 | `omnivggt/models/aggregator.py` | 359–370 | 既有 |
| `ObjectTokenCrossAttentionBlock` | `omnivggt/models/omnivggt.py` | 15–45 | 既有(zero-init gate ⛔ 未啟用) |
| `ObjectQueryPooler` | `omnivggt/heads/object_pose_head.py` | — | 已實作但 ⛔ 關閉(flag off 不建構) |
| Object encoding `_encode_object_prototypes` | `omnivggt/models/omnivggt.py` | 240–329 | 既有 |
| Prototype fusion | `omnivggt/models/omnivggt.py` | 339–370 | 既有 |
| Object Pose Head | `omnivggt/heads/object_pose_head.py` | 116–194 | 既有(baseline 單一 query;query 替換路徑 ⛔ 休眠) |
| Pose Transformer Decoder | `omnivggt/heads/pose_transformer.py` | 303–358 | 既有(attn weights 回傳路徑僅 attn loss 才用,⛔ 未啟用) |
| Object Mask Head | `omnivggt/heads/object_mask_head.py` | 10–68 | 既有 |
| Loss | `omnivggt/loss.py` | 46–571 | `attn_mask_loss` 已實作但 ⛔ 未啟用 |

---

## 4. 資料流(training forward pass,本 config = no_pooler baseline)

```
Input Batch:
├─ images        : (B, S=50, 3, 518, 476)   scene 多視角影像
├─ object_images : (B, S_obj=4, 3, H, W)    object 參考視角 (top/front/back/bottom)
└─ object_id     : (B,)                       物件 ID (caching 用)

OmniVGGT.forward():
│
├─ [OBJECT ENCODING] (frozen encoder, no_grad)
│  ├─ object_aggregator.embed_images(object_images)          → DINOv2 patch embed
│  ├─ forward_from_patch_tokens(return_layer_tokens=True)    → 抽 layer {4,11,17,23}
│  └─ 完整 object patch tokens flatten 當 cross-attn K/V(no_pooler)
│       (⛔ Query Pooler 關閉:layer-23 tokens 不再壓成 object queries)
│
├─ [SCENE AGGREGATION + OBJECT FUSION]
│  ├─ aggregator.embed_images(images)
│  └─ forward_from_patch_tokens(layer_postprocessor=progressive_object_fusion)
│     └─ for layer 0..23:
│        ├─ Frame Attention   (B·S, P, C)
│        ├─ Global Attention  (B, S·P, C)
│        └─ if layer in {4,11,17,23}:
│           └─ x = x + ObjectTokenCrossAttn(Q=scene, K/V=obj tokens)
│              (⛔ zero-init gate 未啟用,直接殘差相加)
│
├─ [OBJECT MASK HEAD]  Conv C→256→128→1 + 上採樣 → object_mask (B,S,H,W)(不變)
│
└─ [OBJECT POSE HEAD]  baseline(單一 learnable query)
   ├─ K/V = flatten 全部 scene patch tokens (B, S·P, C)
   ├─ Q   = 1 個 learnable query (TransformerDecoder num_tokens=1, token_dim=1)
   ├─ TransformerDecoder(depth=6, heads=8) + IEF ×1
   │   (⛔ attn_mask_loss 未啟用,不抽取 cross-attn weights)
   ├─ decpose      → rot6d (symmetric_rot6d)
   ├─ dectranslate → translation (3D)
   ├─ decsize      → size_log (3D)
   └─ presence_branch → presence logit

LOSS:
   └─ object_mask_loss / object_presence_loss / object_srt_loss
      (⛔ attn_mask_loss、⛔ relative_pose_loss 皆未啟用)
```

---

## 5. 重點設計說明

### 5.1 雙編碼器 (two-tower) — `freeze_object_encoder = True`(不變)
- **Scene 分支**: 可訓練的 VGGT aggregator (`self.aggregator`)。
- **Object 分支**: 一份 **凍結的 bf16 拷貝** (`self.object_aggregator`),只把 4 張參考圖
  編碼成 object tokens,不回傳梯度。
- **為什麼**: 若共用權重,scene aggregator 訓練時 object 編碼會 step-by-step 飄移;
  獨立凍結拷貝讓 object encoding 全程固定。代價是 backbone 參數記憶體約翻倍
  (用 bf16 存可砍半:~3.74GB fp32 → ~1.87GB bf16)。

### 5.2 漸進式 cross-attention 注入(⛔ 本 config 不開 zero-init gate)
在 aggregator 第 `4, 11, 17, 23` 層後各插入一個 `ObjectTokenCrossAttentionBlock`
(16 heads):scene patch 當 query,object tokens 當 key/value,輸出**直接殘差相加**。

**為什麼本實驗關掉 gate**(`object_cross_attn_zero_gate=False`):gate 把每個注入點輸出
乘 `tanh(α_l)`、α_l 初始 0,適用於 cross-attn 區塊「新初始化」的情況。但本實驗 warm-start
自 `checkpoint-1-688`,其 cross-attn 區塊**已經訓練過**;若此時設 α=0,反而會在 step 0
把學到的注入貢獻歸零,與「保護 warm-start 特徵」的初衷相反。故只有在 warm-start 自
cross-attn 全新初始化的 checkpoint 時才建議啟用。

### 5.3 `no_pooler`(K/V)+ ⛔ Query Pooler(Q,已還原關閉)
- **K/V 端(不變)**:cross-attn 直接用完整 object patch tokens,保留最多匹配細節。
- ⛔ **Q 端(已關閉,`enable_object_query_pooler=False`)**:曾規劃用 32 個 learnable
  seeds 對 layer-23 object tokens 做 attention pooling,得到物體條件化的 pose-decoder
  queries `(B, 32, 1024)`。實驗顯示對 pose 精度沒有幫助,已關閉;`ObjectQueryPooler`
  模組在 flag off 時不會被建構,pose head 退回單一 learnable query。程式碼保留,
  重新啟用只需把 flag 設回 True。
- **記憶體備註**:no_pooler 下 K/V 長度 ≈ 4×P_obj(≈5k tokens),4 個注入層 × S=50
  的 scene,若記憶體吃緊可對 K/V 做 2× 空間降採樣(strided conv)。

### 5.4 Pose head — baseline 單一 query + IEF
`context_pool="flatten"`、6 層 decoder、IEF ×1 的原始流程。decoder 用單一 learnable
query(`num_tokens=1, token_dim=1`)解出 pose,後接 decpose / dectranslate / decsize /
presence_branch。object 資訊全靠 aggregator 4 層 cross-attn 注入 scene tokens,pose head
本身不直接吃 object queries(query 替換路徑已 ⛔ 關閉,見 §5.3)。

### 5.5 ⛔ Attn mask loss — 本實驗未啟用
`enable_attn_mask_loss=False`。此 loss 的設計(供參考):用 GT mask 在 patch 解析度直接
監督 pose decoder 的 cross-attention map,把 focus 從「期望」變成「約束」。本 config 採
coverage 形式(in-mask attention mass 的 `-log`,見 config 註解),而非舊文件的 Dice+BCE。
本實驗先跑 query 替換的對照,確認穩定後再考慮加上 attention 監督(見 §8 消融順序)。

---

## 6. 啟用 / 關閉的元件總表

| 元件 | 狀態 | config flag |
|------|------|-------------|
| Object Mask Head | ✅ 啟用 | `enable_object_mask=True` |
| Object Presence | ✅ 啟用 | `enable_object_presence=True` |
| Object SRT (Pose) Head | ✅ 啟用(baseline 單一 query) | `enable_object_srt=True` |
| Multi-layer object cross-attn | ✅ 啟用(無 gate) | `enable_multi_layer_object_prototype_cross_attn=True` |
| Frozen object encoder (bf16) | ✅ 啟用 | `freeze_object_encoder=True`, `freeze_object_encoder_bf16=False` |
| Object prototype pooler(K/V 端) | ❌ 關閉 | `disable_object_prototype_pooler=True` |
| Query pooler(pose head Q 端) | ⛔ 關閉(已還原) | `enable_object_query_pooler=False` |
| Pose query 聚合 | ⛔ 不生效 | `object_query_aggregation="attention_pool"`(僅 pooler 開啟時用) |
| Attn mask loss | ⛔ 關閉 | `enable_attn_mask_loss=False`(weight 0.5, coverage) |
| Cross-attn zero-init gate | ⛔ 關閉 | `object_cross_attn_zero_gate=False` |
| Relative-pose loss (SMOC-Net) | ⛔ 關閉 | `enable_relative_pose_loss=False` |
| Depth head | ❌ 關閉 | `enable_depth=False` |
| Point head | ❌ 關閉 | `enable_point=False` |
| Camera head | ❌ 關閉 | `enable_camera=False` |
| Patch embed (DINOv2) | 🔒 凍結 | `patch_embed_freeze=True` |

---

## 7. 關鍵超參數

| 項目 | 值 |
|------|-----|
| Backbone | DINOv2 ViT-L/14, embed_dim=1024, 24 層 |
| Cross-attn 注入層 | (4, 11, 17, 23) |
| Cross-attn heads | 16 |
| Zero-init gate | ⛔ 關閉(`object_cross_attn_zero_gate=False`) |
| Object queries | ⛔ 關閉(`enable_object_query_pooler=False`,pose head 用單一 learnable query) |
| Pose transformer | depth=6, heads=8, dim=1024, mlp_dim=1024, dim_head=64 |
| IEF iters | 1 |
| Pose 表示 | symmetric_rot6d (對稱感知) |
| attn_mask_loss | ⛔ 關閉(實作為 coverage, weight 0.5) |
| train_batch_images | 50 |
| resolution | (518, 476) |
| object 參考視角 | 4 (ids 0,5,8,19 = top/front/back/bottom) |
| Warm-start | checkpoint-1-688/model.safetensors |

---

## 8. 實作順序與消融建議

逐步疊加,讓每個改動的貢獻可拆解:

0. **Baseline ◀ 本 config(`..._no_pooler_revert`)**:no_pooler + 單一 learnable query,
   無 attn loss、無 gate。以下改動目前皆**已還原關閉**。
1. **Step 1 — Query 替換(⛔ 已試,無幫助 → 還原)**:Query Pooler + pose decoder query
   換成 object queries。實驗對照 `no_pooler` baseline 後確認 pose 沒有提升,故關閉。
   重啟方式:`enable_object_query_pooler=True`。
2. **Step 2 — Attn mask loss(⛔ 尚未啟用)**:加 attention 監督,觀察 attn map 是否落在
   物體上、pose 精度變化。啟用方式:`enable_attn_mask_loss=True`。
3. **Step 3(視需要)— Zero-init gate(⛔ 尚未啟用)**:本實驗 warm-start 自已訓練過
   cross-attn 的 checkpoint,故不適用;僅在 warm-start 自全新初始化 cross-attn、且訓練
   初期 loss 不穩時才考慮。
4. **記憶體優化(獨立)**:K/V 2× 降採樣,僅在 OOM 或吞吐不足時啟用。

## 9. 待確認 / TODO

- ⛔ Query Pooler 已試過、無幫助而還原;若日後重啟,待確認單層 attention pooling 是否
  足夠(或需 2 層 + FFN)、32 queries 聚合方式(attention pooling vs. learnable [pose] token)。
- attn_mask_loss(尚未啟用)監督「第 1 層」cross-attn 是否最佳;可試對全部 6 層平均監督。
- attn map 在 S=50 中物體不可見 frame 的行為,需與 presence logit 的一致性檢查。
- 4 個參考視角(top/front/back/bottom)是否足以涵蓋物體外觀,待評估。
