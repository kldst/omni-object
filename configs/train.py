# ======================================================
# OmniVGGT Training Configuration
# ======================================================

# accelerate launch --num_processes=2 train_omnivggt.py --config configs/train.py

# == Common Configuration ==
output_dir = "outputs"
exp_name = "0419_omnivggt_trajectory"
logging_dir = "logs"

# == Logging Configuration ==
wandb = True
tensorboard = False
report_to = "tensorboard"
num_save_log = 1
num_save_visual = 10000
checkpointing_steps = 2000

# == Model Configuration ==
model_url = "/omni_vggt/omnivggt_pretrain_model/OmniVGGT.safetensors"
# model_url = "/mnt/train-data-4-hdd/yian/freepose/omni-object/omnivggt_pretrain_model/OmniVGGT.safetensors"
model_load_strict = False
model_requires_grad = False
patch_embed_freeze = True
load_patch_embed_from_hub = False
patch_embed_pretrained_path = None
enable_point = False
enable_depth = True
enable_camera = False
enable_object_srt = True
object_srt_head_freeze = False
enable_multi_layer_object_prototype_cross_attn = True
object_cross_attn_freeze = False
object_prototype_poolers_freeze = False
object_prototype_layer_indices = (4, 11, 17, 23)
object_prototype_num_tokens = 32
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
num_train_epochs = 100
gradient_accumulation_steps = 1
max_grad_norm = 1.0
debug_print_object_paths = False
debug_print_object_paths_steps = 1
debug_print_object_paths_max_samples = 1
cam_drop_prob = 0.1
depth_drop_prob = 0.0
always_use_depth_gt = True
save_each_epoch = False


# == Optimizer Configuration ==
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.05

# == Learning Rate Configuration ==
lr = 1e-4
lr_patch_embed = 1e-4
lr_camera_head = 1e-4
lr_depth_head = 1e-4
lr_point_head = 1e-4
lr_object_srt_head = 1e-4
lr_object_cross_attn = 1e-4
lr_object_prototype_poolers = 1e-4

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 0
eta_min_factor = 1e-4  # Minimum learning rate factor for cosine decay

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

# == Visualization Configuration ==
save_glb_visualization = False
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
train_batch_images = 60
num_workers = 0
resolution = (518, 518)

train_dataset = (
    "SixDPose("
    "dataset_location='/dataset/0419_trajectory_test_4scene_500frame', "
    "OBJECT_INPUT_ROOT='/dataset/object_space_renders_all', "
    "dset='train', "
    "scene_num_views=1, "
    "object_input_views=(1, 5, 10, 15), "
    "only_scene_start='scene_0000', "
    "only_scene_end='scene_0005', "
    "verify_files=True, "
    "z_far=20, "
    "resolution=(518, 518), "
    "transform=ColorJitter, "
    "seed=42)"
)

                    
                    
              
