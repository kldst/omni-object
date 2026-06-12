# OmniVGGT 模型架構 — `train_hc_diverse24.py` (0612, no_pooler)

> 本文件對應 config: [`configs/train_hc_diverse24.py`](../../configs/train_hc_diverse24.py)
> 實驗名稱: `hc_only_diverse24_14k_0612_no_pooler`
> 任務: HouseCat6D camera-frame 6D 物件姿態估計

---

## 1. 概觀

這是一個 **two-tower(雙編碼器)6D 物件姿態估計** 架構,基於 VGGT
(Visual Geometry Grounded Transformer)aggregator:

1. **DINOv2 ViT-L/14 backbone** 從 scene 影像與 object 參考影像抽 patch 特徵。
2. **VGGT aggregator**(24 層,frame/global 交替注意力)融合多視角 scene 資訊。
3. **多層 object cross-attention**(layer 4/11/17/23)把 object 特徵漸進式注入 scene。
4. **三個 head** 解碼最後一層 scene tokens:**mask / presence / pose(SRT)**。

> ⚠️ 本 config 把 **depth / point / camera** 三個 head 全部關閉
> (`enable_depth/point/camera = False`,對應 loss weight 也都是 0)。
> 並且 `disable_object_prototype_pooler = True`,所以 32-token pooler **未啟用**
> (這就是實驗名稱 `no_pooler` 的由來)。

---

## 2. 架構圖

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
                          │   DINOv2 ViT-L/14 patch embed → 24 層                 │
                          │   torch.no_grad() (object_encoder_no_grad=True)       │
                          └───────────────────────┬─────────────────────────────┘
                                                  │ 抽取 layer {4,11,17,23} 的
                                                  │ object patch tokens
                                                  │ (disable_pooler=True →
                                                  │  直接 flatten 原始 tokens,
                                                  │  不經過 32-token pooler)
                                                  ▼
                            object_prototypes_by_idx  =  {4:…, 11:…, 17:…, 23:…}
                                                  │   (每層一組, 當 cross-attn 的 K/V)
                                                  │
   ┌──────────────────────┐                      │
   │ scene images         │                      │
   │ (B, S=50, 3,518,476) │                      │
   └──────────┬───────────┘                      │
              ▼                                   │
   ┌──────────────────────────┐                  │
   │ DINOv2 ViT-L/14 PatchEmbed│  (patch_embed_freeze=True, 凍結)
   │ + camera/register tokens  │                  │
   └──────────┬───────────────┘                  │
              ▼                                   │
   ╔══════════════════════════════════════════╗  │
   ║   VGGT Aggregator  (24 層, 可訓練)         ║  │
   ║   交替注意力:                              ║  │
   ║   Frame-Attn  (B·S, P, C)  幀內            ║  │
   ║   Global-Attn (B, S·P, C)  跨幀            ║  │
   ║                                            ║  │
   ║   ┌─ layer 4  ──► [Object Cross-Attn] ◄────╫──┘  ← Q=scene patch
   ║   ├─ layer 11 ──► [Object Cross-Attn] ◄────╫──┐    K/V=object protos
   ║   ├─ layer 17 ──► [Object Cross-Attn] ◄────╫──┤    16 heads
   ║   └─ layer 23 ──► [Object Cross-Attn] ◄────╫──┘    (漸進式注入物件資訊)
   ╚══════════════════════╤═════════════════════╝
                          │  最後一層 aggregated tokens  (B, S, P, C)
                          │
        ┌─────────────────┼─────────────────────────────┐
        ▼                                                ▼
 ┌─────────────────────┐                  ┌──────────────────────────────────┐
 │  Object Mask Head   │                  │   Object Pose Head (SRT)          │
 │  Conv: C→256→128→1  │                  │   context_pool = "flatten"        │
 │  雙線性上採樣到原圖  │                  │   ┌────────────────────────────┐ │
 │                     │                  │   │ TransformerDecoder         │ │
 │  → object_mask      │                  │   │ depth=6, heads=8, dim=1024 │ │
 │    (B,S,H,W)        │                  │   │ + IEF 迭代 ×1              │ │
 └─────────────────────┘                  │   └─────────────┬──────────────┘ │
                                          │     ┌───────────┼───────────┐    │
   (depth / point / camera                │     ▼           ▼           ▼    │
    heads 在此 config                     │  decpose   dectranslate  decsize │
    全部 DISABLED)                        │  rot6d(6)   trans(3)    log-size(3)│
                                          │     │                            │
                                          │     └─ presence_branch → logit   │
                                          └──────────────────────────────────┘

 LOSS (全部加權相加):
   object_mask_loss      = BCE + Dice          (weight 1.0)
   object_presence_loss  = BCE                 (weight 1.0)
   object_srt_loss       = L1(pose 6D + trans + size), symmetric_rot6d, 對稱感知
```

---

## 3. 元件對照表

| 元件 | 檔案 | 行數 |
|------|------|------|
| 主模型 `OmniVGGT` | `omnivggt/models/omnivggt.py` | 79–565 |
| 建構子 | `omnivggt/models/omnivggt.py` | 80–207 |
| 模型載入 `load_model` | `train_utils.py` | 229–350 |
| Aggregator (VGGT) | `omnivggt/models/aggregator.py` | 25–462 |
| `ZeroAggregator` | `omnivggt/models/omnivggt_aggregator.py` | 19–450 |
| Layer postprocessor 注入點 | `omnivggt/models/aggregator.py` | 359–370 |
| `ObjectTokenCrossAttentionBlock` | `omnivggt/models/omnivggt.py` | 15–45 |
| `ObjectPrototypePool` (本 config 未用) | `omnivggt/models/omnivggt.py` | 48–76 |
| Object encoding `_encode_object_prototypes` | `omnivggt/models/omnivggt.py` | 240–329 |
| Prototype fusion | `omnivggt/models/omnivggt.py` | 339–370 |
| Object Pose Head | `omnivggt/heads/object_pose_head.py` | 116–194 |
| Pose Transformer Decoder | `omnivggt/heads/pose_transformer.py` | 303–358 |
| Object Mask Head | `omnivggt/heads/object_mask_head.py` | 10–68 |
| Loss | `omnivggt/loss.py` | 46–571 |

---

## 4. 資料流(training forward pass)

來源: `train_omnivggt.py` 370–414

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
│  └─ disable_pooler=True → 直接 flatten 原始 object patch tokens 當 K/V
│
├─ [SCENE AGGREGATION + OBJECT FUSION]
│  ├─ aggregator.embed_images(images)
│  └─ forward_from_patch_tokens(layer_postprocessor=progressive_object_fusion)
│     └─ for layer 0..23:
│        ├─ Frame Attention   (B·S, P, C)
│        ├─ Global Attention  (B, S·P, C)
│        └─ if layer in {4,11,17,23}:
│           └─ ObjectTokenCrossAttn(Q=scene patch, K/V=object tokens, 16 heads)
│
├─ [OBJECT MASK HEAD]  Conv C→256→128→1 + 上採樣 → object_mask (B,S,H,W)
│
└─ [OBJECT POSE HEAD]  flatten scene patches → TransformerDecoder(6 層) + IEF×1
   ├─ decpose      → rot6d (symmetric_rot6d)
   ├─ dectranslate → translation (3D)
   ├─ decsize      → size_log (3D)
   └─ presence_branch → presence logit
```

---

## 5. 重點設計說明

### 5.1 雙編碼器 (two-tower) — `freeze_object_encoder = True`
- **Scene 分支**: 可訓練的 VGGT aggregator (`self.aggregator`)。
- **Object 分支**: 一份 **凍結的 bf16 拷貝** (`self.object_aggregator`),只把 4 張參考圖
  編碼成 object tokens,不回傳梯度。
- **為什麼**: 若共用權重,scene aggregator 訓練時 object 編碼會 step-by-step 飄移;
  獨立凍結拷貝讓 object encoding 全程固定。代價是 backbone 參數記憶體約翻倍
  (用 bf16 存可砍半:~3.74GB fp32 → ~1.87GB bf16)。
- 結構建立: `omnivggt.py:133–140`;複製權重 + 凍結: `train_utils.py:331–343`。

### 5.2 漸進式 cross-attention 注入(核心)
在 aggregator 第 `4, 11, 17, 23` 層後各插入一個 `ObjectTokenCrossAttentionBlock`
(16 heads):scene patch 當 query,object tokens 當 key/value,逐層把「物件長相」
注入場景特徵。透過 aggregator 的 `layer_postprocessor` callback 機制接入。

### 5.3 `disable_object_prototype_pooler = True`(no_pooler)
原設計用 `ObjectPrototypePool` 把 object patch 壓成 32 個 prototype token;
本 config 關閉它,cross-attn 直接用 frozen encoder 輸出的**完整 object patch tokens**
flatten 後當 K/V。`object_prototype_num_tokens=32` 在此模式下被忽略。
代價:context 從 32 變成 `S_obj*P_obj`,較重,但中間沒有 learned compression。

### 5.4 Pose head — IEF 迭代回歸
`object_pose_context_pool="flatten"` 把所有 scene patch 攤平成單一序列,丟進
6 層 Transformer decoder,用 learnable query token + IEF (`ief_iters=1`) 迭代回歸殘差。
輸出 rot6d / translation / size_log / presence。

---

## 6. 啟用 / 關閉的元件總表

| 元件 | 狀態 | config flag |
|------|------|-------------|
| Object Mask Head | ✅ 啟用 | `enable_object_mask=True` |
| Object Presence | ✅ 啟用 | `enable_object_presence=True` |
| Object SRT (Pose) Head | ✅ 啟用 | `enable_object_srt=True` |
| Multi-layer object cross-attn | ✅ 啟用 | `enable_multi_layer_object_prototype_cross_attn=True` |
| Frozen object encoder (bf16) | ✅ 啟用 | `freeze_object_encoder=True`, `freeze_object_encoder_bf16=True` |
| Object prototype pooler (32 tokens) | ❌ 關閉 | `disable_object_prototype_pooler=True` |
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
| Pose transformer | depth=6, heads=8, dim=1024, mlp_dim=1024, dim_head=64 |
| IEF iters | 1 |
| Pose 表示 | symmetric_rot6d (對稱感知) |
| train_batch_images | 50 |
| resolution | (518, 476) |
| object 參考視角 | 4 (ids 0,5,8,19 = top/front/back/bottom) |
| Warm-start | checkpoint-1-688/model.safetensors |
