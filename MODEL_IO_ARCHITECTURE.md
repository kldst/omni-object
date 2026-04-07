# OmniVGGT Model Input / Output Architecture

這份文件根據以下程式碼整理：

- `omnivggt/models/omnivggt.py`
- `omnivggt/models/omnivggt_aggregator.py`
- `omnivggt/models/aggregator.py`
- `omnivggt/heads/camera_head.py`
- `omnivggt/heads/dpt_head.py`
- `omnivggt/heads/head_act.py`
- `omnivggt/utils/pose_enc.py`
- `omnivggt/utils/rotation.py`
- `visual_util.py`
- `inference.py`

## 1. 整體結論

`OmniVGGT` 的對外介面可以分成兩層：

1. `OmniVGGT.forward(...)`
2. `OmniVGGT.inference(...)`

這兩個 API 的核心輸入模態相同：

- RGB images
- optional camera extrinsics
- optional camera intrinsics
- optional depth
- optional depth mask

差別在於：

- `forward(...)` 主要給 training，用隨機方式選擇哪些 frame 的 camera / depth 當成 GT 注入。
- `inference(...)` 主要給推論，用 `camera_gt_index` / `depth_gt_index` 明確指定哪些 frame 有已知 camera / depth。

模型的主要輸出是：

- `pose_enc`: 相機 pose encoding
- `depth`: 深度圖
- `depth_conf`: 深度信心
- `world_points`: 每個 pixel 的 3D world point
- `world_points_conf`: 3D point 信心
- `images`: 原始輸入影像

注意：模型本體直接輸出的是 `pose_enc`，不是 `extrinsic` / `intrinsic`。  
`extrinsic` 和 `intrinsic` 是在 `inference.py` 中用 `pose_encoding_to_extri_intri(...)` 後處理解碼出來的。

## 2. 對外 API: Model Input 結構

### 2.1 `OmniVGGT.forward(...)`

定義位置：`omnivggt/models/omnivggt.py`

```python
def forward(
    self,
    images: torch.Tensor,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
    depth: torch.Tensor = None,
    mask: torch.Tensor = None,
):
```

#### 一般化 tensor shape

- `images`: `[B, S, 3, H, W]`，或官方也接受 `[S, 3, H, W]`
- `extrinsics`: `[B, S, 3, 4]`
- `intrinsics`: `[B, S, 3, 3]`
- `depth`: `[B, S, H, W, 1]`
- `mask`: `[B, S, H, W]`

其中：

- `B` = batch size
- `S` = frame 數量 / sequence length
- `H, W` = 影像高寬

如果 `images` 是 4 維 `[S, 3, H, W]`，程式會自動 `unsqueeze(0)` 變成 `[1, S, 3, H, W]`。  
官方 demo 就是用這個路徑。

### 2.2 `OmniVGGT.inference(...)`

```python
def inference(
    self,
    images: torch.Tensor,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
    depth: torch.Tensor = None,
    mask: torch.Tensor = None,
    depth_gt_index: list = None,
    camera_gt_index: list = None,
):
```

比 `forward(...)` 多兩個 index list：

- `depth_gt_index: List[int]`
- `camera_gt_index: List[int]`

它們表示：

- 哪些 frame 有真實 depth，可注入模型
- 哪些 frame 有真實 camera，可注入模型

在 inference 模式下，模型不再隨機抽樣，而是直接依照這兩個 index list 注入。

### 2.3 官方 loader 產生的實際 input 形狀

`visual_util.load_images_and_cameras(...)` 會回傳：

- `images`: `[S, 3, H, W]`
- `extrinsics`: `[1, S, 3, 4]`
- `intrinsics`: `[1, S, 3, 3]`
- `depthmaps`: `[1, S, H, W, 1]`
- `masks`: `[1, S, H, W]`
- `depth_indices`: `List[int]`
- `camera_indices`: `List[int]`

這代表官方 pipeline 預設其實是單 batch (`B=1`)。

### 2.4 每個 input 的語意

#### `images`

- RGB image
- `ImgNorm = ToTensor()`，所以值域是 `[0, 1]`
- 進到 aggregator 後，還會再做一次 ResNet mean/std normalization

#### `extrinsics`

- shape: `[B, S, 3, 4]`
- world-to-camera
- OpenCV convention
- loader 讀入 `.txt` 時原始是 camera-to-world，之後會反轉成 world-to-camera

#### `intrinsics`

- shape: `[B, S, 3, 3]`
- pixel unit 相機內參矩陣
- loader 會依 resize / crop 同步調整 `fx, fy, cx, cy`

#### `depth`

- shape: `[B, S, H, W, 1]`
- 缺少 depth 的 frame 會補全零
- 有效深度需要 `> 1e-5`

#### `mask`

- shape: `[B, S, H, W]`
- 對應 depth 的有效區域 mask
- 有 depth 的地方通常為 `1`
- 沒有 depth 的 frame 會是全零

#### `camera_gt_index`

- 例如 `[0, 2, 5]`
- 表示第 0、2、5 張影像有 GT camera 可以注入

#### `depth_gt_index`

- 例如 `[1, 4]`
- 表示第 1、4 張影像有 GT depth 可以注入

## 3. 輸入前處理與尺寸規則

官方 `load_images_and_cameras(...)` 的規則如下：

- 固定 `new_width = target_size`，預設 `518`
- `new_height` 依比例縮放，並四捨五入到 14 的倍數
- 若 `new_height > target_size`，會做中心裁切到 `target_size`

因此在官方流程中通常滿足：

- `W = 518`
- `H <= 518`
- `H` 與 `W` 都可被 patch size `14` 整除

所以 patch token 數量為：

```text
P = (H / 14) * (W / 14)
```

若輸入是正方形 518 x 518，則：

```text
P = 37 * 37 = 1369
```

## 4. `omnivggt` 內部 token 架構

這部分主要在 `ZeroAggregator`。

### 4.1 RGB token

影像先進 `patch_embed`，得到：

- `patch_tokens`: `[B*S, P, 1024]`

其中 `embed_dim = 1024`。

### 4.2 Special tokens

Aggregator 額外建立兩種 token：

- `camera_token`: 每個 frame 1 個，shape `[B*S, 1, 1024]`
- `register_token`: 每個 frame 4 個，shape `[B*S, 4, 1024]`

因此 patch token 的起始 index 是：

```text
patch_start_idx = 1 + 4 = 5
```

### 4.3 Camera modality 注入

如果某些 frame 有 GT camera：

1. 取出該 frame 的 `extrinsics` 與 `intrinsics`
2. 將 extrinsics 做 normalization
3. 轉成 pose encoding
4. 用線性層投影到 1024 維
5. 注入到 camera token

camera pose encoding 的 shape 是：

- `[B, S_cam, 9]`

9 維內容為：

- `[:3]`: translation `T`
- `[3:7]`: quaternion，順序是 `X, Y, Z, W`，其中 `W` 是 scalar-last
- `[7]`: vertical FoV
- `[8]`: horizontal FoV

### 4.4 Depth modality 注入

如果某些 frame 有 GT depth：

1. 先用 mask 對 depth 做 normalize
2. 把 `normalized_depth` 和 `mask` 串成 2-channel tensor
3. 經過 `depth_patch_embed`
4. 得到與 RGB patch token 對齊的 depth tokens

depth 注入前的 2-channel tensor shape：

- `[B * S_depth, 2, H, W]`

depth token shape：

- `[B*S, P, 1024]`

### 4.5 Token 串接後的結構

注入後的 token 序列為：

```text
[camera token (1), register tokens (4), patch tokens (P)]
```

總 shape：

- `tokens`: `[B*S, P + 5, 1024]`

### 4.6 Alternating attention 後的 aggregated tokens

模型會交替做：

- frame attention
- global attention

預設 `depth = 24` 且 `aa_block_size = 1`，所以最後會得到：

- `aggregated_tokens_list`: 長度 `24`

每一層的 tensor shape 是：

- `[B, S, P + 5, 2048]`

為什麼是 `2048`：

- frame branch 輸出 `[B, S, P+5, 1024]`
- global branch 輸出 `[B, S, P+5, 1024]`
- 兩者最後在 channel 維度 concat

## 5. 對外 API: Model Output 結構

`OmniVGGT.forward(...)` 與 `OmniVGGT.inference(...)` 都會回傳一個 `predictions: dict`。

### 5.1 直接輸出的 keys

#### `predictions["pose_enc"]`

- shape: `[B, S, 9]`
- 這是最後一次 iteration 的 camera pose encoding

9 維語意：

- `0:3` -> translation `T`
- `3:7` -> quaternion `X, Y, Z, W`
- `7` -> vertical FoV
- `8` -> horizontal FoV

#### `predictions["pose_enc_list"]`

- type: `list[Tensor]`
- 長度預設是 `4`
- 每個元素 shape: `[B, S, 9]`

這是 `CameraHead` 的 iterative refinement 中，每次 iteration 的 pose prediction。

#### `predictions["depth"]`

- shape: `[B, S, H, W, 1]`
- 由 `depth_head = DPTHead(output_dim=2, activation="exp")` 輸出
- 第一個 channel 經 activation 後變成正值深度

#### `predictions["depth_conf"]`

- shape: `[B, S, H, W]`
- 由 `depth_head` 的最後一個 channel 經 `1 + exp(conf)` 產生

#### `predictions["world_points"]`

- shape: `[B, S, H, W, 3]`
- 由 `point_head = DPTHead(output_dim=4, activation="inv_log")` 輸出
- 代表每個 pixel 在 world frame 的 3D 座標

#### `predictions["world_points_conf"]`

- shape: `[B, S, H, W]`
- point map 對應的 confidence

#### `predictions["images"]`

- shape: `[B, S, 3, H, W]`
- 原始輸入影像，保留在 output dict 內方便後續 visualization

### 5.2 模型沒有直接輸出，但官方 inference 會補上的 keys

在 `inference.py` 裡，會進一步做：

```python
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    images.shape[-2:]
)
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic
```

因此 demo pipeline 後面常看到的額外欄位是：

#### `predictions["extrinsic"]`

- shape: `[B, S, 3, 4]`
- 由 `pose_enc` 解碼而來

#### `predictions["intrinsic"]`

- shape: `[B, S, 3, 3]`
- 由 `pose_enc` 裡的 FoV 反推出來
- principal point 被固定設在影像中心 `(W/2, H/2)`

## 6. 各 head 的實際輸出 shape 推導

### 6.1 Camera head

`CameraHead` 從最後一層 `aggregated_tokens_list[-1]` 中取：

- `tokens[:, :, 0]`

也就是每個 frame 的第 0 個 token，當作 camera token。

所以 camera branch 是：

```text
[B, S, P+5, 2048]
 -> 取 token index 0
 -> [B, S, 2048]
 -> iterative refinement
 -> [B, S, 9]
```

### 6.2 Depth head

`depth_head = DPTHead(output_dim=2)`，經過 `activate_head(...)` 後：

- 前 1 個 channel 變成 depth
- 最後 1 個 channel 變成 confidence

因此實際輸出是：

- `depth`: `[B, S, H, W, 1]`
- `depth_conf`: `[B, S, H, W]`

### 6.3 Point head

`point_head = DPTHead(output_dim=4)`，經過 `activate_head(...)` 後：

- 前 3 個 channel 變成 XYZ
- 最後 1 個 channel 變成 confidence

因此實際輸出是：

- `world_points`: `[B, S, H, W, 3]`
- `world_points_conf`: `[B, S, H, W]`

## 7. 訓練與推論在 input 行為上的差異

### training (`forward`)

- `camera_gt_index` 不由外部傳入
- `ZeroAggregator.select_camera_gt(...)` 會隨機選前綴 frames，例如 `[0, 1, 2]`
- `depth_gt_index` 也不由外部傳入
- `ZeroAggregator.select_depth_gt(...)` 會隨機選任意 subset

所以 training 階段會模擬「只有部分 frame 有 camera/depth supervision」的情境。

### inference (`inference`)

- `camera_gt_index` 由外部明確提供
- `depth_gt_index` 由外部明確提供
- 沒有提供的 frame 會使用 zero placeholder，不注入該模態

這就是 OmniVGGT 的「任意組合輔助模態」介面。

## 8. 最精簡的 I/O 摘要

### Input

```python
inputs = {
    "images":          [S, 3, H, W] or [B, S, 3, H, W],
    "extrinsics":      [B, S, 3, 4],
    "intrinsics":      [B, S, 3, 3],
    "depth":           [B, S, H, W, 1],
    "mask":            [B, S, H, W],
    "depth_gt_index":  List[int],   # inference only
    "camera_gt_index": List[int],   # inference only
}
```

### Output

```python
predictions = {
    "pose_enc":           [B, S, 9],
    "pose_enc_list":      List[[B, S, 9]],   # len=4
    "depth":              [B, S, H, W, 1],
    "depth_conf":         [B, S, H, W],
    "world_points":       [B, S, H, W, 3],
    "world_points_conf":  [B, S, H, W],
    "images":             [B, S, 3, H, W],
}
```

### Post-decoded output used by demo

```python
predictions["extrinsic"] = [B, S, 3, 4]
predictions["intrinsic"] = [B, S, 3, 3]
```

## 9. 一句話總結

OmniVGGT 的核心設計是：  
用 RGB 影像作為主輸入，將 optional camera / depth 以 token injection 的方式注入 aggregator，最後同時預測相機 pose encoding、dense depth、以及 dense 3D point map。
