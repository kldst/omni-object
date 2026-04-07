from importlib import import_module

SixDPose = import_module("omnivggt.datasets.6Dpose.6dpose").SixDPose

__all__ = ["SixDPose"]
