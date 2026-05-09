from importlib import import_module

SixDPose = import_module("omnivggt.datasets.6Dpose.6dpose_trajectory_noscale").SixDPose

__all__ = ["SixDPose"]
