# ======================================================
# OmniVGGT Training Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs"
exp_name = "omnivggt-test"
logging_dir = "logs"

# == Logging Configuration ==
wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 500
checkpointing_steps = 100

# == Model Configuration ==
model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model_load_strict = False
model_requires_grad = False
patch_embed_freeze = True
enable_point = False
enable_depth = False
enable_camera = False
enable_object_srt = True
object_srt_head_freeze = False
enable_multi_layer_object_prototype_cross_attn = True
object_cross_attn_freeze = False
object_prototype_poolers_freeze = False
object_prototype_layer_indices = (4, 11, 17, 23)
object_prototype_num_tokens = 4
object_prototype_object_encoder_no_grad = True
object_cross_attn_heads = 16
object_pose_context_pool = "flatten"
object_pose_use_global_scene_object_concat = False
object_pose_transformer_depth = 6
object_pose_transformer_heads = 8
object_pose_transformer_mlp_dim = 1024
object_pose_transformer_dim_head = 64
object_pose_transformer_dropout = 0.0
object_pose_transformer_emb_dropout = 0.0
object_pose_transformer_norm = "layer"
object_pose_transformer_dim = 1024
object_pose_ief_iters = 1
object_pose_init_params_path = None

# == Training Configuration ==
mixed_precision = "bf16"  # Options: "no", "fp16", "bf16"
seed = 42
num_train_epochs = 2
gradient_accumulation_steps = 1
max_grad_norm = 1.0
cam_drop_prob = 0.1
depth_drop_prob = 0.3

# == Optimizer Configuration ==
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration ==
lr = 2e-5
lr_patch_embed = 1e-5
lr_camera_head = 2e-5
lr_depth_head = 2e-5
lr_point_head = 2e-5
lr_object_srt_head = 2e-5
lr_object_cross_attn = 2e-5
lr_object_prototype_poolers = 2e-5

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 50
eta_min_factor = 0.1  # Minimum learning rate factor for cosine decay

# == Loss Configuration ==
# Camera loss
camera_loss_weight = 5.0
camera_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Depth loss
depth_loss_weight = 1.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98

# Point loss
point_loss_weight = 1.0
point_gradient_loss_fn = "normal"
point_valid_range = 0.98

# Object 6D pose loss
object_srt_loss_weight = 1.0
object_srt_loss_type = "l1"
object_srt_weight_pose = 1.0
object_srt_weight_translation = 1.0
object_srt_init_w = 1.0

# == Visualization Configuration ==
vis_conf_threshold = 0.2
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_show_cam = True
vis_mask_sky = False
vis_prediction_mode = "Predicted Depth"

# == Resume Configuration ==
resume_model_path = None

# == Dataset Configuration ==
resolution = (518, 518)

train_dataset = (
    "100 @ SixDPose("
    "dataset_location='/mnt/train-data-4-hdd/yian/6dpose_obj/0405_fixedCam_diffpose_1k', "
    "OBJECT_INPUT_ROOT='/mnt/train-data-4-hdd/yian/6dpose_obj/0316_fixedCam_1k/object_space_rgb', "
    "dset='test', "
    "selected_views=(1,), "
    "object_input_views=(1, 3, 4), "
    "only_run_start='run_1000', "
    "only_run_end='run_1099', "
    "verify_files=True, "
    "z_far=20, "
    "resolution=(518, 518), "
    "transform=ColorJitter, "
    "seed=42)"
)
