from src.algorithms.common.pixel_control import PixelControlAlgorithm
from src.algorithms.spr.spr import SPRAgent
from src.algorithms.spr.spr import SPRConfig
from src.algorithms.spr.spr import SRSPRAgent
from src.algorithms.spr.spr import SRSPRConfig


class SPRAlgorithm(PixelControlAlgorithm):
    agent_cls = SPRAgent
    config_cls = SPRConfig
    prefetch_fixed_replay = True


class SRSPRAlgorithm(PixelControlAlgorithm):
    agent_cls = SRSPRAgent
    config_cls = SRSPRConfig

__all__ = ["SPRAlgorithm", "SRSPRAlgorithm"]
