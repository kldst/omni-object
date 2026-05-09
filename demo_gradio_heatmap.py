import argparse
import math
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

# python3 demo_gradio_heatmap.py --checkpoint /mnt/train-data-4-hdd/yian/6dpose_obj/OmniVGGT-official/outputs/0405_omnivggt_single_image_pose_5sameobject/checkpoint-24-6000/model.safetensors
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from demo_gradio_6dpose import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRETRAIN_MODEL,
    build_model_from_config,
    image_path_for_view,
    list_objects_for_run,
    list_runs,
    list_scene_views_for_run,
    load_config,
    load_object_tensor,
    load_scene_inputs,
    object_image_path,
    parse_dataset_ctor_arg,
    resolve_dataset_settings,
    resolve_checkpoint_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_images_to_numpy(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images


def _images_to_gallery(images, prefix: str):
    return [(image, f"{prefix} {idx + 1}") for idx, image in enumerate(images)]


def _infer_patch_grid(num_patches: int):
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(f"Patch count {num_patches} is not a square grid.")
    return side, side


def _compute_heatmap_values(tokens: torch.Tensor):
    return torch.linalg.norm(tokens, dim=-1)


def _compute_relative_delta_values(scene_tokens: torch.Tensor, delta_tokens: torch.Tensor):
    scene_norm = torch.linalg.norm(scene_tokens, dim=-1)
    delta_norm = torch.linalg.norm(delta_tokens, dim=-1)
    return delta_norm / scene_norm.clamp_min(1e-6)


def _compute_cosine_change_values(scene_tokens: torch.Tensor, fused_tokens: torch.Tensor):
    cosine = torch.nn.functional.cosine_similarity(scene_tokens, fused_tokens, dim=-1, eps=1e-6)
    return 1.0 - cosine


def _tokens_to_heatmaps(token_values: np.ndarray, images, prefix: str):
    gallery = []
    for idx, image in enumerate(images):
        heat = token_values[idx]
        h_patch, w_patch = _infer_patch_grid(int(heat.size))
        heat = heat.reshape(h_patch, w_patch).astype(np.float32)
        heat = heat - float(heat.min())
        heat = heat / max(float(heat.max()), 1e-6)
        heat_resized = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat_uint8 = np.clip(255.0 * heat_resized, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)
        gallery.append((overlay, f"{prefix} {idx + 1}"))
    return gallery


def _summarize_tensor(name: str, tensor: torch.Tensor):
    arr = tensor.detach().float().cpu()
    return (
        f"{name}: mean={arr.mean().item():.4f}, std={arr.std().item():.4f}, "
        f"min={arr.min().item():.4f}, max={arr.max().item():.4f}"
    )


def _final_scene_patch_tokens(aggregated_tokens_list, patch_start_idx: int):
    return aggregated_tokens_list[-1][:, :, patch_start_idx:, :]


class HeatmapApp:
    def __init__(self, config_path: Path, checkpoint_path: str | None):
        self.cfg = load_config(config_path)
        dataset_settings = resolve_dataset_settings(self.cfg)
        self.dataset_root = Path(dataset_settings["dataset_root"])
        self.object_root = Path(dataset_settings["object_root"])
        self.object_views = tuple(int(v) for v in dataset_settings["object_input_views"])
        self.resolution = tuple(int(v) for v in dataset_settings["resolution"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint_path)
        self.model = build_model_from_config(self.cfg, self.checkpoint_path, self.device)
        self.run_choices = list_runs(self.dataset_root)
        if not self.run_choices:
            raise RuntimeError(f"No runs found under {self.dataset_root / 'out_image'}")
        self.available_checkpoints = self.discover_checkpoints()

    def discover_checkpoints(self):
        candidates = set()
        if DEFAULT_PRETRAIN_MODEL.is_file():
            candidates.add(str(DEFAULT_PRETRAIN_MODEL))
        model_url = Path(str(self.cfg.get("model_url", ""))).expanduser()
        if model_url.is_file():
            candidates.add(str(model_url))
        for path in sorted((PROJECT_ROOT / "outputs").glob("**/model.safetensors")):
            candidates.add(str(path))
        return sorted(candidates)

    def load_checkpoint(self, checkpoint_path: str):
        resolved = Path(checkpoint_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        self.model = build_model_from_config(self.cfg, resolved, self.device)
        self.checkpoint_path = resolved
        if str(resolved) not in self.available_checkpoints:
            self.available_checkpoints = sorted([*self.available_checkpoints, str(resolved)])
        return (
            f"Loaded checkpoint: `{self.checkpoint_path}`",
            gr.update(choices=self.available_checkpoints, value=str(self.checkpoint_path)),
            str(self.checkpoint_path),
        )

    def get_object_choices(self, run_name: str):
        return list_objects_for_run(self.dataset_root, run_name)

    def get_scene_view_choices(self, run_name: str):
        return list_scene_views_for_run(self.dataset_root, run_name)

    def input_galleries(self, run_name: str, object_name: str, scene_view: int):
        scene_paths = [image_path_for_view(self.dataset_root, run_name, int(scene_view))]
        object_paths = [object_image_path(self.object_root, object_name, view_idx) for view_idx in self.object_views]
        return (
            _images_to_gallery(_load_images_to_numpy(scene_paths), "Scene"),
            _images_to_gallery(_load_images_to_numpy(object_paths), "Object"),
        )

    def _collect_token_visualizations(self, scene_images, object_images, depth_tensor, mask_tensor, use_depth_input: bool):
        scene_images = self.model._ensure_batched_images(scene_images)
        object_images = self.model._ensure_batched_images(object_images)

        depth_gt_index = [0] if use_depth_input else []
        depth_arg = depth_tensor if use_depth_input else None
        mask_arg = mask_tensor if use_depth_input else None

        scene_aggregated_tokens_list, scene_patch_start_idx = self.model.aggregator.inference(
            images=scene_images,
            extrinsics=None,
            intrinsics=None,
            depth=depth_arg,
            mask=mask_arg,
            depth_gt_index=depth_gt_index,
            camera_gt_index=[],
        )
        scene_patch_tokens = _final_scene_patch_tokens(scene_aggregated_tokens_list, scene_patch_start_idx)

        if not self.model.enable_multi_layer_object_prototype_cross_attn:
            raise RuntimeError("Heatmap visualization requires enable_multi_layer_object_prototype_cross_attn=True")

        object_prototypes_by_idx, object_patch_tokens = self.model._encode_object_prototypes(object_images)

        def progressive_object_fusion(layer_idx, scene_layer_tokens, scene_layer_patch_start_idx):
            return self.model._apply_progressive_object_prototype_cross_attention(
                layer_idx,
                scene_layer_tokens,
                scene_layer_patch_start_idx,
                object_prototypes_by_idx,
            )

        fused_tokens_list, fused_patch_start_idx = self.model.aggregator.inference(
            images=scene_images,
            extrinsics=None,
            intrinsics=None,
            depth=depth_arg,
            mask=mask_arg,
            depth_gt_index=depth_gt_index,
            camera_gt_index=[],
            layer_postprocessor=progressive_object_fusion,
        )
        fused_scene_patch_tokens = _final_scene_patch_tokens(fused_tokens_list, fused_patch_start_idx)
        delta_tokens = fused_scene_patch_tokens - scene_patch_tokens

        return {
            "scene_patch_tokens": scene_patch_tokens[0],
            "object_patch_tokens": object_patch_tokens[0],
            "fused_scene_patch_tokens": fused_scene_patch_tokens[0],
            "delta_scene_patch_tokens": delta_tokens[0],
        }

    def run_heatmap(self, run_name: str, object_name: str, scene_view: int, use_depth_input: bool):
        if not run_name:
            raise ValueError("Please select a run.")
        if not object_name:
            raise ValueError("Please select an object.")
        scene_view = int(scene_view)
        use_depth_input = bool(use_depth_input)

        scene_gallery, object_gallery = self.input_galleries(run_name, object_name, scene_view)
        scene_images = [item[0] for item in scene_gallery]
        object_images_np = [item[0] for item in object_gallery]
        scene_tensor, depth_tensor, mask_tensor = load_scene_inputs(
            self.dataset_root,
            run_name,
            scene_view,
            self.resolution,
            self.device,
        )
        object_tensor = load_object_tensor(
            self.object_root,
            object_name,
            self.object_views,
            self.resolution,
            self.device,
        )

        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=self.device.type == "cuda"):
                token_dict = self._collect_token_visualizations(
                    scene_tensor,
                    object_tensor,
                    depth_tensor,
                    mask_tensor,
                    use_depth_input=use_depth_input,
                )

        scene_norm = _compute_heatmap_values(token_dict["scene_patch_tokens"]).cpu().numpy()
        object_norm = _compute_heatmap_values(token_dict["object_patch_tokens"]).cpu().numpy()
        fused_norm = _compute_heatmap_values(token_dict["fused_scene_patch_tokens"]).cpu().numpy()
        delta_norm = _compute_heatmap_values(token_dict["delta_scene_patch_tokens"]).cpu().numpy()
        relative_delta = _compute_relative_delta_values(
            token_dict["scene_patch_tokens"], token_dict["delta_scene_patch_tokens"]
        ).cpu().numpy()
        cosine_change = _compute_cosine_change_values(
            token_dict["scene_patch_tokens"], token_dict["fused_scene_patch_tokens"]
        ).cpu().numpy()

        log_msg = "\n".join(
            [
                f"checkpoint={self.checkpoint_path}",
                f"run={run_name}, object={object_name}, scene_view={scene_view}, object_views={self.object_views}",
                f"use_depth_input={use_depth_input}",
                _summarize_tensor("scene_patch_tokens", token_dict["scene_patch_tokens"]),
                _summarize_tensor("object_patch_tokens", token_dict["object_patch_tokens"]),
                _summarize_tensor("fused_scene_patch_tokens", token_dict["fused_scene_patch_tokens"]),
                _summarize_tensor("delta_scene_patch_tokens", token_dict["delta_scene_patch_tokens"]),
                (
                    "relative_delta: "
                    f"mean={float(relative_delta.mean()):.4f}, std={float(relative_delta.std()):.4f}, "
                    f"min={float(relative_delta.min()):.4f}, max={float(relative_delta.max()):.4f}"
                ),
                (
                    "cosine_change: "
                    f"mean={float(cosine_change.mean()):.4f}, std={float(cosine_change.std()):.4f}, "
                    f"min={float(cosine_change.min()):.4f}, max={float(cosine_change.max()):.4f}"
                ),
            ]
        )

        return (
            log_msg,
            scene_gallery,
            object_gallery,
            _tokens_to_heatmaps(scene_norm, scene_images, "Scene"),
            _tokens_to_heatmaps(object_norm, object_images_np, "Object"),
            _tokens_to_heatmaps(fused_norm, scene_images, "Fused"),
            _tokens_to_heatmaps(delta_norm, scene_images, "Delta"),
            _tokens_to_heatmaps(relative_delta, scene_images, "Relative Delta"),
            _tokens_to_heatmaps(cosine_change, scene_images, "Cosine Change"),
        )


def build_demo(app: HeatmapApp):
    default_run = app.run_choices[0]
    default_objects = app.get_object_choices(default_run)
    default_object = default_objects[0] if default_objects else None
    default_scene_views = app.get_scene_view_choices(default_run)
    default_scene_view = default_scene_views[0] if default_scene_views else 1

    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.HTML(
            f"""
            <h1>OmniVGGT Token Heatmap Viewer</h1>
            <p>Visualize baseline scene tokens, object tokens, prototype-fused scene tokens,
            token delta, relative delta, and cosine change for the current OmniVGGT object-pose setup.</p>
            <p><code>config={DEFAULT_CONFIG_PATH}</code><br/>
            <code>default_pretrain_model={DEFAULT_PRETRAIN_MODEL}</code></p>
            """
        )

        checkpoint_status = gr.Markdown(f"Loaded checkpoint: `{app.checkpoint_path}`")
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=app.available_checkpoints,
                value=str(app.checkpoint_path),
                label="Checkpoint Presets",
                allow_custom_value=True,
            )
            checkpoint_textbox = gr.Textbox(value=str(app.checkpoint_path), label="Checkpoint Path")
            load_model_button = gr.Button("Load Model")

        with gr.Row():
            run_dropdown = gr.Dropdown(choices=app.run_choices, value=default_run, label="Run")
            object_dropdown = gr.Dropdown(choices=default_objects, value=default_object, label="Object")
            scene_view_dropdown = gr.Dropdown(choices=default_scene_views, value=default_scene_view, label="Input Scene View")
            use_depth_checkbox = gr.Checkbox(value=True, label="Use Depth Input")
            generate_btn = gr.Button("Generate Heatmaps", variant="primary")

        log_output = gr.Markdown("Select a run/object/view, then click Generate Heatmaps.")

        with gr.Tabs():
            with gr.Tab("Input Views"):
                scene_input_gallery = gr.Gallery(label="Scene Input", columns=1, height="420px", object_fit="contain", preview=True)
                object_input_gallery = gr.Gallery(label="Object Inputs", columns=len(app.object_views), height="260px", object_fit="contain", preview=True)
            with gr.Tab("Scene Tokens"):
                scene_token_gallery = gr.Gallery(label="Scene Token Norm Heatmap", columns=1, height="520px", object_fit="contain", preview=True)
            with gr.Tab("Object Tokens"):
                object_token_gallery = gr.Gallery(label="Object Token Norm Heatmap", columns=len(app.object_views), height="320px", object_fit="contain", preview=True)
            with gr.Tab("Fused Scene Tokens"):
                fused_scene_gallery = gr.Gallery(label="Prototype-Fused Scene Heatmap", columns=1, height="520px", object_fit="contain", preview=True)
            with gr.Tab("Delta"):
                delta_scene_gallery = gr.Gallery(label="Token Delta Heatmap", columns=1, height="520px", object_fit="contain", preview=True)
            with gr.Tab("Relative Delta"):
                relative_delta_gallery = gr.Gallery(label="Relative Delta Heatmap", columns=1, height="520px", object_fit="contain", preview=True)
            with gr.Tab("Cosine Change"):
                cosine_change_gallery = gr.Gallery(label="Cosine Change Heatmap", columns=1, height="520px", object_fit="contain", preview=True)

        def refresh_run_controls(run_name: str):
            objects = app.get_object_choices(run_name)
            object_value = objects[0] if objects else None
            scene_views = app.get_scene_view_choices(run_name)
            scene_view_value = scene_views[0] if scene_views else None
            return (
                gr.update(choices=objects, value=object_value),
                gr.update(choices=scene_views, value=scene_view_value),
            )

        def refresh_inputs(run_name: str, object_name: str, scene_view):
            if not run_name or not object_name or scene_view is None:
                return [], []
            return app.input_galleries(run_name, object_name, int(scene_view))

        def sync_checkpoint_path(selected_value: str):
            return selected_value

        run_dropdown.change(refresh_run_controls, inputs=run_dropdown, outputs=[object_dropdown, scene_view_dropdown])
        run_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        object_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        scene_view_dropdown.change(refresh_inputs, inputs=[run_dropdown, object_dropdown, scene_view_dropdown], outputs=[scene_input_gallery, object_input_gallery])
        checkpoint_dropdown.change(sync_checkpoint_path, inputs=checkpoint_dropdown, outputs=checkpoint_textbox)
        load_model_button.click(app.load_checkpoint, inputs=checkpoint_textbox, outputs=[checkpoint_status, checkpoint_dropdown, checkpoint_textbox])
        generate_btn.click(
            app.run_heatmap,
            inputs=[run_dropdown, object_dropdown, scene_view_dropdown, use_depth_checkbox],
            outputs=[
                log_output,
                scene_input_gallery,
                object_input_gallery,
                scene_token_gallery,
                object_token_gallery,
                fused_scene_gallery,
                delta_scene_gallery,
                relative_delta_gallery,
                cosine_change_gallery,
            ],
        )

        if default_object is not None:
            demo.load(
                refresh_inputs,
                inputs=[run_dropdown, object_dropdown, scene_view_dropdown],
                outputs=[scene_input_gallery, object_input_gallery],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio heatmap demo for OmniVGGT")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    app = HeatmapApp(args.config, args.checkpoint)
    demo = build_demo(app)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.dataset_root), str(app.object_root)],
    )


if __name__ == "__main__":
    main()
