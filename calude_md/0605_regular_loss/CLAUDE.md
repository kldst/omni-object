# 0605 — Relative-Pose Regularization Loss (SMOC-Net style)

## 目標
參考 [SMOC-Net](../../../SMOC-Net) 的 `relative_camera_loss`,在 HouseCat6D 訓練中加入
**「同一靜止物體、跨視角的相對 pose 一致性」** 作為 regularization loss:

> 同一個物體實例在兩個不同相機視角下,模型預測的**相對物體旋轉**應該等於兩視角間的**相對相機旋轉**。

訓練 config: [`configs/train_hc_diverse24.py`](../../configs/train_hc_diverse24.py)(HouseCat6D-only)。

## 為什麼這對「全監督」仍有用
SMOC-Net 是**自監督**(沒有物體 pose GT),用相對相機 pose 當唯一幾何錨。我們是**全監督**,
所以這個 loss 不是補標籤,而是 **跨視角一致性 regularizer**。最大價值在**對稱 / category-level**:
per-frame 的 symmetric loss 允許每個 view 各自挑一個對稱分支 → 跨 view 會「跳分支」抖動;
relative-pose loss(配合 symmetry-aware min)會強制兩個 view 落在**同一個相對轉換**上,壓低多視角抖動。

## 關鍵幾何發現(和最初設想不同)
HouseCat6D 的 dataset 把 `extrinsic` **寫死成單位矩陣**
([`housecat6d_camera_pose.py:367`](../../omnivggt/datasets/housecat6d/housecat6d_camera_pose.py#L367)),
物體 pose 直接存在相機系(`r_native_to_cam`, `t_cam`)。
**所以不能從 `batch["extrinsic"]` 取相對相機 pose。**

改用 **從 GT 物體 pose 反推**:物體靜止,model→world 旋轉 `R_{m2w}` 對兩 view 相同,
且 `R^o = R_{cam←world} · R_{m2w}`,因此
```
R^o_i · (R^o_j)^T = R_{cam_i←world} · R_{cam_j←world}^T   (= 相對相機旋轉,與物體絕對朝向無關)
```
HouseCat6D 的對齊 `R_align`(category 常數)在這個乘積裡會自然抵消
(`R^o = R_native·R_align^T` → `R^o_i (R^o_j)^T = R_native_i R_native_j^T`),所以用 aligned 或 native 都一致。

## 座標 / 編碼注意事項
- **rot6d 編碼**:head 輸出的 `object_pose` 受 [`compute_object_srt_loss`](../../omnivggt/loss.py#L230)
  直接以 `_rotation_matrix_to_rot6d(R) = R[:, :2].reshape(6)`(row-major 前兩欄)監督。
  因此新增的 `_rot6d_to_matrix` 必須是它的**精確逆**:`col0 = d6[[0,2,4]]`, `col1 = d6[[1,3,5]]` → Gram-Schmidt。
- **translation 是 depth-mean 正規化過的**(每個 frame 用各自的 `object_translation_scale`)。
  旋轉一致性不受影響;若要做 translation 一致性需先用 `object_translation_scale` 還原 metric(預設關閉)。

## 對稱問題怎麼解(重點)
對稱物體下,兩個 view 可能各自落在不同對稱分支 `S_a, S_b`,使預測相對旋轉偏掉一個對稱:
```
R^o_i,pred (R^o_j,pred)^T ≈ R^o_i,gt · (S_a S_b^T) · (R^o_j,gt)^T,   S_a S_b^T ∈ 對稱群
```
所以候選集合 = `{ R^o_i,gt · S · (R^o_j,gt)^T : S ∈ sym }`,loss 取**對候選的最小測地距離**。
- 對稱資訊沿用 [`mixed_symmetry_info.json`](../../mixed_symmetry_info.json),key 格式 `HouseCat6DCameraPose:<object_id>`
  (連續對稱已離散成 `symmetry_continuous_steps` 個,沿用 object_srt 的設定)。
- 非對稱物體 sym = {I} → 退化成單純測地距離。
- **translation 不受旋轉對稱影響**(對稱軸過物體中心),故 translation 項不做 symmetry 處理。

## 要改的檔案
1. **[`omnivggt/loss.py`](../../omnivggt/loss.py)**
   - 新增 `_rot6d_to_matrix`(`_rotation_matrix_to_rot6d` 的逆)。
   - 新增 `_geodesic_angle(R_a, R_b)`(`arccos((trace(AᵀB)-1)/2)`,夾 clamp)。
   - 新增 `compute_object_relative_pose_loss(...)`:把 batch reshape 成 pair (B//2, 2),
     **self-check**(同 `object_id`+`dataset`、兩者都 `has_object`)濾出合法 pair,
     算 symmetry-aware 相對旋轉 loss(+ 選配相對 translation loss)。
   - `MultitaskLoss.__init__` 加 `relative_pose=None`;`forward` 加分支(在 object_srt 之後)。
2. **[`train_utils.py`](../../train_utils.py)** `build_loss_criterion`:依 `relative_pose_loss_weight` 組 `relative_pose` dict。
3. **[`omnivggt/datasets/housecat6d/housecat6d_camera_pose.py`](../../omnivggt/datasets/housecat6d/housecat6d_camera_pose.py)**
   - `__init__` 加 `relative_pose_pairing=False`, `pair_min_frame_gap=0`。
   - 加 `build_pair_groups()`:依 `(scene_name, object_name)` 分組,只保留 ≥2 frame 的組。
   - `make_sampler` 在開啟 pairing 時回傳 `PairedObjectBatchSampler`(讓 batch 內 `(2k,2k+1)` 為同實例不同 frame)。
4. **[`omnivggt/datasets/base/batched_sampler.py`](../../omnivggt/datasets/base/batched_sampler.py)**:新增 `PairedObjectBatchSampler`。
5. **[`configs/train_hc_diverse24.py`](../../configs/train_hc_diverse24.py)**:加 loss 權重 + 在 train_dataset 字串開 `relative_pose_pairing=True`。
6. **[`test_housecat6d_camera_pose.py`](../../omnivggt/datasets/housecat6d/test_housecat6d_camera_pose.py)**:新增 `--test-relative-pose` 模式驗證配對 + loss。

## 安全性設計
- **loss 內 self-check**:即使 pairing 沒開或 batch 順序被打亂,不合法 pair 會被 mask 掉 → loss 退化成 0,不會污染訓練。
- 預設 `relative_pose_loss_weight = 0.1`(小,當 regularizer),`weight_rot=1.0`, `weight_trans=0.0`(translation 先關)。
- 單卡(`accelerate launch --num_processes=1`)→ sampler 順序直接對應 batch 順序,pair 不會被 accelerate 重切。

## 驗收標準(self-verify)
- `_rot6d_to_matrix(_rotation_matrix_to_rot6d(R)) == R`(round-trip)。
- 用 **GT 當預測**餵 loss → `loss_rel_rot ≈ 0`;擾動預測 → loss 明顯 > 0。
- 對稱物體:把預測 = GT 右乘一個對稱 `S` → symmetry-aware loss 仍 ≈ 0。
- 配對 sampler 產出的相鄰 pair:`object_name` 相同、`image_id` 不同、batch 大小為偶數且整除。
- dataset 既有測試(overlay)不被破壞。

## 狀態(全部完成並通過驗證)
- [x] 計畫文件
- [x] loss.py — `_rot6d_to_matrix` / `_geodesic_angle` / `compute_object_relative_pose_loss` + `MultitaskLoss` 分支
- [x] train_utils.py — `build_loss_criterion` 加 `relative_pose`(weight=0 自動關閉)
- [x] dataset + sampler — `build_pair_groups()` + `PairedObjectBatchSampler` + `make_sampler` 分支
- [x] config — `train_hc_diverse24.py` 加權重 + train_dataset 開 `relative_pose_pairing=True, pair_min_frame_gap=20`
- [x] test script — `--test-relative-pose` 模式
- [x] 驗證

## 驗證結果(`freepose` conda env)
跑 `python omnivggt/datasets/housecat6d/test_housecat6d_camera_pose.py --test-relative-pose --dset train --scene-glob 'scene01' --batch-size 8 --pair-min-frame-gap 20`:
- rot6d↔matrix round-trip 誤差 `1.8e-7` ✅
- 配對 sampler:首個 batch 4 對全部「同 scene+object、不同 frame」,0 個 invalid ✅
- GT 當預測 → `loss_rel_rot = 0.082°`(arccos clamp 數值地板,等同 0)✅;view0 擾動 20° → `10.04°` ✅
- 對稱物(object_id=56,72 候選):兩 view 各右乘對稱 → `loss_rel_rot = 0.082°`(對稱被吸收)✅
- 端到端:`build_loss_criterion` + `MultitaskLoss.forward` objective 有限、relative 分支可開可關 ✅
- 反向傳播:loss 對 `object_pose` 梯度有限、norm 正常(7.54° 誤差 → grad_norm 0.71)✅

## 注意:GT-as-prediction 的 0.082° 地板
`_geodesic_angle` 的 `arccos` 在 `cos→1` 用 `eps=1e-6` clamp,地板 `≈sqrt(2·eps)=1.4e-3 rad=0.08°`。
完美預測時這個常數無梯度、可忽略(對 objective 貢獻 `0.1·1.4e-3≈1.4e-4`)。若在意可把 eps 調更小。

## 啟用/調參
- 開關:`configs/train_hc_diverse24.py` 的 `relative_pose_loss_weight`(預設 0.1;設 0 完全關閉,連 loss 分支都不建)。
- 只約束旋轉:`relative_pose_weight_trans=0.0`(預設)。要開 translation 一致性設 >0(會用 `object_translation_scale` 還原 metric)。
- 配對只在 **train_dataset** 開;val_dataset 沒開,val 的 relative loss 因 self-check 會退化成 ~0(安全但不具參考性)。
- `pair_min_frame_gap`:避免挑到幾乎相同的視角(預設 20 frame)。
- ⚠️ `train_batch_images` 必須是**偶數**(目前 60,OK)。

---

# 追加 (0605 第二批):Object Cache + Benchmark Validation

## A. Object encoder cache(加速 train/val)
凍結的 reference-image ViT 編碼是 object encoding 最貴的部分,且只依賴凍結 encoder → 可跨 batch/epoch cache。
- [omnivggt.py](../../omnivggt/models/omnivggt.py):`_compute_object_layer_tokens`(貴,可 cache)/ `_gather_object_layer_tokens_cached`(LRU + batch 內去重)/ `_encode_object_prototypes(object_images, object_ids)`。**只 cache encoder 的 layer tokens;可訓練的 poolers 照常 live 跑**(梯度不受影響)。
- `forward`/`inference` 加 `object_ids` 參數;[train_omnivggt.py](../../train_omnivggt.py) 在 `_prepare_batch_and_compute_loss` 傳 `batch['object_id']`。
- **前置條件(全部滿足才正確)**:`object_prototype_object_encoder_no_grad=True`、object encoder 凍結、reference 圖確定性。
- **ref 去增強**:[housecat6d_camera_pose.py](../../omnivggt/datasets/housecat6d/housecat6d_camera_pose.py) 加 `object_ref_color_jitter=False`(預設),object refs 從 `ColorJitter` 改用 `ImgNorm`(scene 圖增強不受影響)。
  - ⚠️ 名詞澄清:此 codebase 的 `ImgNorm = Compose([ToTensor()])`,**只有 ToTensor**(PIL→tensor、像素縮到 [0,1]、HWC→CHW),**沒有**減均值/除標準差。`ColorJitter` 則是 `RandomApply(ColorJitter)+RandomGrayscale+ToTensor`。兩者結尾都有 ToTensor,**唯一差別是前面那段隨機增強**。換成 ImgNorm = 「拿掉隨機增強、只留固定 ToTensor」→ 同一張 ref 每次輸出相同 → 編碼確定 → cache 正確。整條 pipeline 其實沒有做 ImageNet 式 normalize,輸入是 [0,1] raw 像素。
- config:`object_encode_cache=True`、`object_encode_cache_max=256`。
- **配對加成**:一個 pair 兩個 sample 是同物體 → 同 batch 內 encoder 只跑一次。
- 驗證:cache on vs off 組出的 tokens **完全相同**;batch 內去重(2 unique rows)、跨 batch 重用、LRU 上限都通過。

## B. Validation 改用官方 benchmark(多卡 scene 分片)
- [housecat_benchmark.py](../../housecat_benchmark.py)(新檔):
  - `run_benchmark_inference(...scenes, output_dir)`:只對「這個 rank 分到的 scene shard」做推論 + 寫 per-frame pkl(不評估)。多 rank 並行安全(各寫各自 scene 子目錄)。
  - `evaluate_and_collect(out_dir, all_scenes)`:讀**全部** pkl,包 VI-Net `compute_independent_mAP`,**抓回傳的 `iou_3d_aps`/`pose_aps`** 取 overall 列回傳 dict。
  - `run_housecat_benchmark(...)`:單進程便利包裝(全 scene 推論 + 評估),給獨立 smoke / 單卡用。
- [train_omnivggt.py](../../train_omnivggt.py) `run_housecat_benchmark_validation`(**路 B,多卡分片**):
  - 每個 DDP rank 用 `scenes[rank::world]` 拿 disjoint shard,用 `unwrap_model` 的 model(各自 GPU)跑推論寫 pkl;
  - `accelerator.wait_for_everyone()` barrier 等所有 rank 寫完;
  - **rank0** `evaluate_and_collect(out_dir, all_scenes)` 讀全部 pkl 算 mAP、log `val_bench/{iou_25,iou_50,iou_75,pose_5deg_2cm,pose_5deg_5cm,pose_10deg_2cm,pose_10deg_5cm}`;再一個 barrier。
  - `num_processes=1` 時自動退化成單卡(rank0 拿全部 scene,barrier 為 no-op)。
  - 設計重點:**所有 rank 都會進到 benchmark 與 barrier**(periodic val 呼叫沒被 is_main_process gate);分到 0 scene 的 rank 也照樣等 barrier → 不 deadlock。比「rank0-only、其餘 idle」快 ~N 倍,且不易踩 NCCL barrier timeout。
- config:`validation_mode="benchmark"`(可切回 `"loss"`)、`benchmark_scenes`(全 5 test scene)、`benchmark_frame_stride=5`、`benchmark_batch_size`、`benchmark_limit`。
- **驗證**:
  - 單卡:`outputs/0521/model.safetensors` 跑 test_scene1(41 frame),抽取 dict 與官方 `eval_housecat_official.py --eval-only` 在同一批 pkl 的 overall 數字**完全一致**。
  - **多卡(2 GPU,`accelerate launch --num_processes=2 --gpu_ids=0,1`)**:rank0→test_scene1、rank1→test_scene2 並行推論,barrier 後 rank0 gather 到 `num_frames=6`(兩 scene 都讀到)→ **分片 + barrier + 檔案 gather 正確**;兩 rank 乾淨結束。(mAP 0.0 同樣是舊 checkpoint × diverse24 不匹配,非 bug。)

### `eval_housecat_official.py`(獨立 eval)vs 訓練內 benchmark(路 B)
- `eval_housecat_official.py` 的多卡是 **subprocess 編排**(`orchestrate`:每 GPU 開獨立 subprocess + `CUDA_VISIBLE_DEVICES`,各自 `build_model(checkpoint)`,跑完父進程讀全部 pkl)。**無 NCCL**,適合**訓練外**對已存 checkpoint 做 eval。
- **不能**在 DDP 訓練中照搬 subprocess 那套(GPU 已被訓練佔住會搶卡 OOM)。路 B 用**現有 DDP rank** 做 scene 分片 + 檔案 gather,達到同樣的「分片並行 + gather」效果但不另起進程。

## C. Batch dump 驗證工具(test script `--dump-batch`)
[test_housecat6d_camera_pose.py](../../omnivggt/datasets/housecat6d/test_housecat6d_camera_pose.py) 新增 `--dump-batch`:
**走真正的 DataLoader(同 sampler + `_intersection_collate`)** 抓一個 batch,再用 `_decollate_item` 把每個 sample 拆回來,
逐項輸出:scene 圖疊上 GT pose(3D bbox + XYZ 軸)、4 張 object reference 圖(縮圖貼在下方)、以及一份 JSON(所有路徑 + pose 值 + has_object)。
- 用途:肉眼確認「object 圖 ↔ scene 圖 ↔ pose ↔ 路徑」在 batch 內對應正確;開 `--relative-pose-pairing` 還會印 pair check(相鄰兩項應同物體不同 frame)。
- collate 細節:str 欄位 → 長度 B 的 list(取 `[i]`);`object_rgb_paths` 被 default_collate **轉置**成「n_views 個 tuple、各長 B」(取 `[view[i] for view in value]`)——`_decollate_item` 都處理了。
- 範例:
  ```
  python omnivggt/datasets/housecat6d/test_housecat6d_camera_pose.py --dump-batch --relative-pose-pairing \
    --dset train --scene-glob 'scene01' \
    --object-image-root .../housecat6d_aligned_object_refs_diverse24 \
    --batch-size 8 --dump-num 8 --pair-min-frame-gap 20 \
    --save-dir .../verification_batch --resolution 518 518
  ```
- **驗證**:scene01、batch 8、pairing 開 → 4 對全部同物體不同 frame(pair check OK);抽看 `can-kidney_beans` overlay,3D bbox/軸正確落在該物體上、下方 4 張 ref 也對應同物體 ✅。

## D. Train / Benchmark 的 cache 與 ColorJitter 分流(0605 第三批)
依使用者需求,**訓練**與 **benchmark** 對 object refs 的 augmentation 與 cache 採不同策略:

| | object refs transform | object encoder cache | 理由 |
|---|---|---|---|
| **訓練** | **ColorJitter**(`object_ref_color_jitter=True`,訓練 dataset 字串) | **OFF**(`object_encode_cache=False`) | 還原 ref augmentation;且 colorjit refs 非確定性,本來就不能 cache。cache 關掉也省 GPU 記憶體(batch 60 + activation,16GB 吃緊)。 |
| **benchmark** | **ColorJitter**(`benchmark_object_ref_color_jitter=True`) | **ON**(`benchmark_object_encode_cache=True`,僅 benchmark 期間暫開) | eval 無 backprop activation、同物體跨多 frame → cache 大幅加速。 |

實作:
- dataset `object_transform = ColorJitter if object_ref_color_jitter else ImgNorm`(**與 scene transform 解耦**,benchmark scene 用 ImgNorm 時 refs 仍能 colorjit)。
- benchmark dataset 透過 [eval_housecat_official.py `build_dataset`](../../eval_housecat_official.py) 收 `object_ref_color_jitter`;`run_scene_inference` 的 `model.inference(..., object_ids=...)` 把 object_id 傳進去(cache 才會生效)。
- [train_omnivggt.py `run_housecat_benchmark_validation`](../../train_omnivggt.py):benchmark 前 `base_model.object_encode_cache = benchmark_object_encode_cache` + `clear_object_cache()`;跑完**還原** `prev` 並再 `clear_object_cache()`(釋放記憶體、不影響訓練)。
- ⚠️ **colorjit + cache 的語意**:cache 用 `object_id` 當 key,所以某物體**第一次**被讀到的「那組隨機 colorjit 編碼」會被凍結、之後該物體所有 frame 都重用 → 每物體**一個固定的隨機增強版本**(非每 frame 重抽)。要乾淨 refs 就把 `benchmark_object_ref_color_jitter=False`。

### Object cache 記憶體
- cache 存「per-layer ViT tokens」在 **GPU** 上,約 **~41MB/物體**(4 層 ×4 視角 ×~1263 token ×1024 ×bf16);上界 `min(物體數, object_encode_cache_max=256)`。HouseCat6D ~190 物體 → 滿載 ~8GB,**訓練時 16GB 卡會吃緊 → 所以訓練關 cache**。會增長到看完所有物體後**持平**(非 leak)。若要訓練也開 cache,建議改存 CPU(尚未實作,可再加 `object_encode_cache_device`)。

### 驗證(0605 第三批)
- config 載入:`object_encode_cache=False`、`benchmark_object_encode_cache=True`、`benchmark_object_ref_color_jitter=True`、train_dataset 含 `object_ref_color_jitter=True` ✅
- benchmark 路徑(單卡,colorjit refs + cache ON,test_scene1,12 frame):cache 用到 **10 個物體**(跨 frame 重用)、跑完 `clear_object_cache()` → 0、metrics 正常回傳 ✅(colorjit + cache 共存不崩)

## 注意事項 / TODO 提醒
- benchmark 用的 `view_ids` / `object_image_root` 會自動取 config 的 `fixed_object_view_ids`(0,5,8,19)與 `housecat6d_object_image_root`(diverse24),與訓練一致。
- benchmark 較重:全 5 scene × frame_stride=5。要更快可調大 `benchmark_frame_stride` 或設 `benchmark_limit`;頻率由 `val_epoch_freq` 控制。
- benchmark 模式下不需要 `val_dataset`(loss-val);若兩者都想看,把 `validation_mode` 切 `"loss"` 另外手動跑 benchmark。
- benchmark inference 走 `model.inference`,目前**沒傳 object_ids**(不吃 cache);因 benchmark 不常跑,影響小。要的話可在 `run_scene_inference` 補。
- 多 GPU:benchmark 只在 main process 跑(其餘 `wait_for_everyone`);目前訓練是 `--num_processes=1`,無虞。

## 跨 dataset 推廣備註
本次只改 HouseCat6D。若要套到 OO9D/REAL275/YCBV:那些 dataset 的 `extrinsic` 若是**真實 w2c**(非單位矩陣),
可以改用 `R_i (R^o_i) ... ` 直接從相機 extrinsic 算相對相機 pose;但目前 loss 用「GT 物體 pose 反推」的版本對兩種情況都成立(因為靜止物體下兩者等價),所以**可直接重用**,只需在各自 dataset 加同樣的 `build_pair_groups` + paired sampler。
