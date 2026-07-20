from src.algorithms.bbf.bbf import BBFAgent
from src.algorithms.bbf.bbf import BBFConfig
from src.algorithms.common.pixel_control import PixelControlAlgorithm


class BBFAlgorithm(PixelControlAlgorithm):
    agent_cls = BBFAgent
    config_cls = BBFConfig

__all__ = ["BBFAlgorithm"]
