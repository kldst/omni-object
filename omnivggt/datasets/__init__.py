from importlib import import_module

from .arkitscenes_high import ARKitScenesHigh
from .bedlam import Bedlam
from .blendedmvs import BlendedMVS
from .co3d import Co3d
from .dl3dv import Dl3dv
from .dynamic_replica import Dynamic_Replica
from .hypersim import Hypersim
from .kubric import Kubric
from .mapfree import MapFree
from .megadepth import MegaDepth
from .mp3d import Mp3d
from .mvs_synth import Mvs_Synth
from .scannet import Scannet
from .scannetppv2 import Scannetppv2
from .spring import Spring
from .tartanair import TarTanAirDUSt3R
from .uasol import Uasol
from .unreal4k import Unreal4k
from .vkitti import Vkitti
from .waymo import Waymo
from .wildrgb import Wildrgb

SixDPose = import_module("omnivggt.datasets.6Dpose.6dpose_trajectory_noscale").SixDPose
OV9DCameraPose = import_module("omnivggt.datasets.6Dpose.ov9d_camera_pose").OV9DCameraPose
OO9DCameraPose = import_module("omnivggt.datasets.oo9d.oo9d_camera_pose").OO9DCameraPose
OO9DGeneratedMultiCameraPose = import_module(
    "omnivggt.datasets.oo9d.oo9d_multi_camera_pose"
).OO9DGeneratedMultiCameraPose
OO9DSingleCameraPose = import_module(
    "omnivggt.datasets.oo9d.oo9d_single_camera_pose"
).OO9DSingleCameraPose
HouseCat6DCameraPose = import_module(
    "omnivggt.datasets.housecat6d.housecat6d_camera_pose"
).HouseCat6DCameraPose
YCBVCameraPose = import_module(
    "omnivggt.datasets.ycbv.ycbv_camera_pose"
).YCBVCameraPose
Real275CameraPose = import_module(
    "omnivggt.datasets.real275.real275_camera_pose"
).Real275CameraPose
Omni6DPoseCameraPose = import_module(
    "omnivggt.datasets.omni6dpose.omni6dpose_camera_pose"
).Omni6DPoseCameraPose

from omnivggt.datasets.utils.transforms import ImgNorm, ColorJitter


def _intersection_collate(batch):
    """Collate dict samples by intersecting keys across the batch.

    When ConcatDataset mixes datasets that return slightly different dict keys
    (e.g. R_align_ycbv_to_ov9d only exists for YCBV samples), default_collate
    raises KeyError on the diverging keys. We collate only the keys present in
    every sample so per-dataset metadata is silently dropped instead of crashing.
    """
    import torch
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return default_collate(batch)
    if not isinstance(batch[0], dict):
        return default_collate(batch)
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys.intersection_update(sample.keys())
    if not common_keys:
        return {}
    filtered = [{k: sample[k] for k in common_keys} for sample in batch]
    return default_collate(filtered)


def get_data_loader(dataset, batch_size, num_workers=8,
                    shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from omnivggt.datasets.utils.misc import get_world_size, get_rank

    world_size = get_world_size()
    rank = get_rank()
    if isinstance(dataset, str):
        dataset = eval(dataset)
    
    try:
        # Let Accelerate own distributed sharding after `accelerator.prepare(...)`.
        # Passing the real world_size/rank here would shard once in the sampler and
        # then a second time in Accelerate, which halves steps per epoch incorrectly.
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=1,
            rank=0,
            drop_last=drop_last,
        )
    except (AttributeError, NotImplementedError):
        # Accelerate.prepare() will own distributed sharding; avoid sharding
        # here too or steps/epoch get divided by world_size twice.
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
            
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
        drop_last=drop_last,
        collate_fn=_intersection_collate,
        )
    
    return data_loader
