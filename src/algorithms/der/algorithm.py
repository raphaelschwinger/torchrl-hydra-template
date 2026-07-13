from src.algorithms.common.pixel_control import PixelControlAlgorithm
from src.algorithms.der.der import DERAgent
from src.algorithms.der.der import DERConfig


class DERAlgorithm(PixelControlAlgorithm):
    agent_cls = DERAgent
    config_cls = DERConfig

__all__ = ["DERAlgorithm"]
