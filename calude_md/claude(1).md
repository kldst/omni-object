# OmniVGGT 模型架構 — `train_hc_diverse24.py` (0612, no_pooler) + Object-Query 修改版

> 本文件對應 config: [`configs/train_hc_diverse24.py`](../../configs/train_hc_diverse24.py)
> 實驗名稱: `hc_only_diverse24_14k_0612_no_pooler`
> 任務: HouseCat6D camera-frame 6D 物件姿態估計
>
> 🆕 標記者為**本次規劃的修改**(尚未實作),目標:在不裁切 scene image 的前提下,
> 讓 pose head 結構性地 focus 在場景中與目標物體相關的 tokens。

---

## 1. 概觀

這是一個 **two-tower(雙編碼器)6D 物件姿態估計** 架構,基於 VGGT
(Visual Geometry Grounded Transformer)aggregator:

1. **DINOv2 ViT-L/14 backbone** 從 scene 影像與 object 參考影像抽 patch 特徵。
2. **VGGT aggregator**(24 層,frame/global 交替注意力)融合多視角 scene 資訊。
3. **多層 object cross-attention**(layer 4/11/17/23)把 object 特徵漸進式注入 scene。
4. **三個 head** 解碼最後一層 scene tokens:**mask / presence / pose(SRT)**。

🆕 **本次修改的核心想法**:現有 cross-attention 是 scene→object 方向的「廣播式」注入,
背景 tokens 也會拿到物件特徵,沒有機制保證 pose head 聚焦在物體區域;且 pose decoder
的 query 是 generic learnable token,與目標物體無關。修改後:

1. 🆕 **Query pooler**:把 object patch tokens 壓成 32 個 object queries(只給 pose head 用,
   cross-attn 的 K/V 維持 no_pooler 的完整 tokens)。
2. 🆕 **Pose decoder query 替換**:learnable query → 32 個 object-conditioned queries,
   讓 decoder 反向(object→scene)主動撈取物體相關區域特徵;背景 tokens 結構上進不了 pose head。
3. 🆕 **Attn mask loss**:用 GT mask 直接監督 pose decoder 的 cross-attention map,
   focus 從「期望」變成「約束」。
4. 🆕(建議)**Zero-init gate**:4 個注入層的 cross-attn 輸出乘 `tanh(α)`、α 初始為 0,
   保護 warm-start checkpoint 的場景特徵分布。

> ⚠️ 本 config 把 **depth / point / camera** 三個 head 全部關閉
> (`enable_depth/point/camera = False`,對應 loss weight 也都是 0)。
> 並且 `disable_object_prototype_pooler = True`,所以 cross-attn 的 K/V **不經 pooler**
> (`no_pooler`)。🆕 注意:修改版重新啟用 `ObjectPrototypePool` 的「結構」,
> 但出口改接 pose head 的 query,**不接 cross-attn 的 K/V**。

---

## 2. 架構圖(修改版)

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
                                     │                          │
                     (完整 tokens 當 cross-attn K/V)       🆕 layer-23 tokens
                                     │                          │
   ┌──────────────────────┐          │                          ▼
   │ scene images         │          │            ┌──────────────────────────┐
   │ (B, S=50, 3,518,476) │          │            │ 🆕 Query Pooler           │
   └──────────┬───────────┘          │            │ 32 learnable seeds        │
              ▼                      │            │ attention-pool object     │
   ┌──────────────────────────┐      │            │ tokens → (B, 32, 1024)    │
   │ DINOv2 PatchEmbed (凍結)  │      │            └─────────────┬────────────┘
   └──────────┬───────────────┘      │                          │
              ▼                      │                          │ object queries
   ╔══════════════════════════════════════════╗                │
   ║   VGGT Aggregator  (24 層, 可訓練)         ║                │
   ║   Frame-Attn / Global-Attn 交替            ║                │
   ║   ┌─ layer 4  ─► [Obj Cross-Attn ×🆕gate] ◄╫─ Q=scene patch │
   ║   ├─ layer 11 ─► [Obj Cross-Attn ×🆕gate] ◄╫─ K/V=obj tokens│
   ║   ├─ layer 17 ─► [Obj Cross-Attn ×🆕gate] ◄╫─ 16 heads      │
   ║   └─ layer 23 ─► [Obj Cross-Attn ×🆕gate] ◄╫─               │
   ╚══════════════════════╤═════════════════════╝                │
                          │ 最後一層 scene tokens (B, S, P, C)    │
        ┌─────────────────┤                                      │
        ▼                 ▼ (flatten 成 K/V)                     │
 ┌─────────────────┐   ┌──────────────────────────────────────┐  │
 │ Object Mask Head │   │ 🆕 Object Pose Decoder (改)           │◄─┘
 │ Conv C→256→128→1 │   │ TransformerDecoder depth=6, heads=8   │
 │ → object_mask    │   │ Q = 32 object queries (原: learnable) │
 │   (B,S,H,W)      │   │ K/V = flatten scene tokens            │
 └─────────────────┘   │ + IEF ×1                              │
                        │   │                                   │
                        │   ├─🆕 第 1 層 cross-attn weights      │
                        │   │   (B, heads, 32, S·P)             │
                        │   │   → 平均 head/query → reshape      │
                        │   │   → (B, S, H/p, W/p) attn map      │──► 🆕 attn_mask_loss
                        │   ▼                                   │     (Dice + BCE vs GT mask)
                        │ 32 queries → attention pooling         │
                        │  (或 learnable [pose] token 聚合)       │
                        │   ├─ decpose      → rot6d(6)           │
                        │   ├─ dectranslate → trans(3)           │
                        │   ├─ decsize      → size_log(3)        │
                        │   └─ presence_branch → logit           │
                        └──────────────────────────────────────┘

 LOSS (全部加權相加):
   object_mask_loss      = BCE + Dice                  (weight 1.0)
   object_presence_loss  = BCE                         (weight 1.0)
   object_srt_loss       = L1(rot6d + trans + size), symmetric_rot6d
   🆕 attn_mask_loss      = Dice + BCE(attn map vs GT mask, patch 解析度)
                            (建議初始 weight 0.5–1.0,attention 穩定後 decay)
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
| `ObjectTokenCrossAttentionBlock` | `omnivggt/models/omnivggt.py` | 15–45 | 🆕 加 zero-init gate |
| `ObjectPrototypePool` | `omnivggt/models/omnivggt.py` | 48–76 | 🆕 改作 Query Pooler,出口接 pose head |
| Object encoding `_encode_object_prototypes` | `omnivggt/models/omnivggt.py` | 240–329 | 既有 |
| Prototype fusion | `omnivggt/models/omnivggt.py` | 339–370 | 既有 |
| Object Pose Head | `omnivggt/heads/object_pose_head.py` | 116–194 | 🆕 query 替換 + queries 聚合 |
| Pose Transformer Decoder | `omnivggt/heads/pose_transformer.py` | 303–358 | 🆕 回傳第 1 層 cross-attn weights |
| Object Mask Head | `omnivggt/heads/object_mask_head.py` | 10–68 | 既有 |
| Loss | `omnivggt/loss.py` | 46–571 | 🆕 新增 `attn_mask_loss` |

---

## 4. 資料流(training forward pass,修改版)

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
│  ├─ 完整 object patch tokens flatten 當 cross-attn K/V(no_pooler 不變)
│  └─ 🆕 layer-23 tokens → Query Pooler → object_queries (B, 32, 1024)
│       (pooler 本身可訓練;輸入來自 frozen encoder,無梯度回 backbone)
│
├─ [SCENE AGGREGATION + OBJECT FUSION]
│  ├─ aggregator.embed_images(images)
│  └─ forward_from_patch_tokens(layer_postprocessor=progressive_object_fusion)
│     └─ for layer 0..23:
│        ├─ Frame Attention   (B·S, P, C)
│        ├─ Global Attention  (B, S·P, C)
│        └─ if layer in {4,11,17,23}:
│           └─ x = x + tanh(α_l) · ObjectTokenCrossAttn(Q=scene, K/V=obj tokens)
│              🆕 α_l 初始 0(每注入層一個標量;訓練起點等價無注入)
│
├─ [OBJECT MASK HEAD]  Conv C→256→128→1 + 上採樣 → object_mask (B,S,H,W)(不變)
│
└─ [OBJECT POSE HEAD]  🆕 修改版
   ├─ K/V = flatten 全部 scene patch tokens (B, S·P, C)
   ├─ Q   = object_queries (B, 32, 1024)        ← 原本: 1 個 learnable query
   ├─ TransformerDecoder(depth=6, heads=8) + IEF ×1
   │   └─ 🆕 第 1 層 cross-attn weights (B, heads, 32, S·P)
   │       → mean over heads, queries → reshape (B, S, H/p, W/p) = attn_map
   ├─ 32 queries → attention pooling(或 [pose] token)→ 單一 pose 向量
   │   (IEF 殘差迭代作用在聚合後輸出上,流程不變)
   ├─ decpose      → rot6d (symmetric_rot6d)
   ├─ dectranslate → translation (3D)
   ├─ decsize      → size_log (3D)
   └─ presence_branch → presence logit

LOSS:
   ├─ object_mask_loss / object_presence_loss / object_srt_loss(不變)
   └─ 🆕 attn_mask_loss = Dice + BCE(attn_map, GT_mask ↓ 到 patch 解析度)
       · 重用 mask head 的同一份 GT mask
       · S=50 為 per-frame 監督;物體不可見 frame 的 GT mask 全零,
         BCE 自然壓低該 frame attention,與 presence 監督互補
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

### 5.2 漸進式 cross-attention 注入 + 🆕 zero-init gate
在 aggregator 第 `4, 11, 17, 23` 層後各插入一個 `ObjectTokenCrossAttentionBlock`
(16 heads):scene patch 當 query,object tokens 當 key/value。

🆕 **gate**:每個注入點輸出乘 `tanh(α_l)`,α_l 為標量、初始化 0。理由:warm-start
自 `checkpoint-1-688`,gate 讓訓練起點的場景特徵分布與 warm-start 完全一致,
避免初期被未收斂的 cross-attn 衝壞;模型自行學習注入強度。aggregator 本身可訓練,
此項為穩定性建議而非必要——可先不加、觀察 loss 曲線後再決定。

### 5.3 `no_pooler`(K/V)+ 🆕 Query Pooler(Q)— 兩件獨立的事
- **K/V 端(不變)**:cross-attn 直接用完整 object patch tokens,保留最多匹配細節。
- 🆕 **Q 端(新增)**:pose decoder 需要固定數量、物體條件化的 queries。
  32 個 learnable seeds 對 layer-23 object tokens 做一次 attention pooling,
  輸出 `(B, 32, 1024)`。實作上即閒置的 `ObjectPrototypePool` 換出口:接 pose head,不接 K/V。
- **記憶體備註**:no_pooler 下 K/V 長度 ≈ 4×P_obj(≈5k tokens),4 個注入層 × S=50
  的 scene,若記憶體吃緊可對 K/V 做 2× 空間降採樣(strided conv),與 Q 端設計互相獨立。

### 5.4 Pose head — 🆕 object-conditioned queries + IEF
原 `context_pool="flatten"` 與 6 層 decoder、IEF 流程不變;改動只有兩處:
1. learnable query → 32 個 object queries(decoder 反向 object→scene 撈特徵,
   背景 tokens 結構上拿不到進入 pose head 的話語權);
2. decoder 輸出端把 32 個 queries 聚合(attention pooling 或 learnable [pose] token)
   後再進 decpose / dectranslate / decsize / presence_branch。

### 5.5 🆕 Attn mask loss — 把 focus 從期望變約束
cross-attention 的 attention map 是「模型正在看哪裡」的熱度圖。ViT token 位置綁定
(positional encoding + residual 保住 patch 對應),故可直接拿 GT mask 在 patch
解析度上監督:看對地方不罰,看錯就吃 Dice+BCE。權重排程建議:初期 0.5–1.0,
attention 穩定落在物體上後 decay,後期容量留給 pose 精度。

---

## 6. 啟用 / 關閉的元件總表

| 元件 | 狀態 | config flag |
|------|------|-------------|
| Object Mask Head | ✅ 啟用 | `enable_object_mask=True` |
| Object Presence | ✅ 啟用 | `enable_object_presence=True` |
| Object SRT (Pose) Head | ✅ 啟用(🆕 修改版) | `enable_object_srt=True` |
| Multi-layer object cross-attn | ✅ 啟用(🆕 +gate) | `enable_multi_layer_object_prototype_cross_attn=True` |
| Frozen object encoder (bf16) | ✅ 啟用 | `freeze_object_encoder=True`, `freeze_object_encoder_bf16=True` |
| Object prototype pooler(K/V 端) | ❌ 關閉 | `disable_object_prototype_pooler=True` |
| 🆕 Query pooler(pose head Q 端) | 🆕 新增 | 建議 flag: `enable_object_query_pooler=True` |
| 🆕 Attn mask loss | 🆕 新增 | 建議 flag: `enable_attn_mask_loss=True`, `attn_mask_loss_weight=0.5` |
| 🆕 Cross-attn zero-init gate | 🆕 建議 | 建議 flag: `object_cross_attn_zero_gate=True` |
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
| 🆕 Gate 初始值 | α_l = 0(每注入層一個標量) |
| 🆕 Object queries 數量 | 32(來源:layer-23 object tokens) |
| Pose transformer | depth=6, heads=8, dim=1024, mlp_dim=1024, dim_head=64 |
| IEF iters | 1 |
| Pose 表示 | symmetric_rot6d (對稱感知) |
| 🆕 attn_mask_loss weight | 0.5–1.0 起始,後期 decay |
| train_batch_images | 50 |
| resolution | (518, 476) |
| object 參考視角 | 4 (ids 0,5,8,19 = top/front/back/bottom) |
| Warm-start | checkpoint-1-688/model.safetensors |

---

## 8. 🆕 實作順序與消融建議

逐步疊加,讓每個改動的貢獻可拆解:

1. **Step 1 — Query 替換**:Query Pooler + pose decoder query 換成 object queries。
   跑一版與 `no_pooler` baseline 對照,確認 mask / presence metrics 不退步、pose 有無提升。
2. **Step 2 — Attn mask loss**:在 Step 1 之上加 attention 監督,觀察 attn map
   視覺化是否落在物體上、pose 精度變化。
3. **Step 3(視需要)— Zero-init gate**:若 Step 1/2 訓練初期 loss 不穩或
   warm-start 特徵被衝壞,再補 gate。
4. **記憶體優化(獨立)**:K/V 2× 降採樣,僅在 OOM 或吞吐不足時啟用。

## 9. 待確認 / TODO

- Query Pooler 用單層 attention pooling 是否足夠,或需 2 層 + FFN。
- 32 queries 的聚合方式:attention pooling vs. learnable [pose] token,需小規模對照。
- attn_mask_loss 監督「第 1 層」cross-attn 是否最佳;可試對全部 6 層平均監督。
- attn map 在 S=50 中物體不可見 frame 的行為,需與 presence logit 的一致性檢查。
- 4 個參考視角(top/front/back/bottom)是否足以涵蓋物體外觀,待評估。
