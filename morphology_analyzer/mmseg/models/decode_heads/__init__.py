# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .hv_head import HVHead
from .cellpose_head import CellPoseHead
from .hv2_head import HV2Head
from .pointnu_head import PointNuHead
from .discriminator import DomainDiscriminator
# from .rpn_head import RPNHead
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

__all__ = [
    "FCNHead",
    "PSPHead",
    "ASPPHead",
    "UPerHead",
    "DepthwiseSeparableASPPHead",
    "DAHead",
    "DLV2Head",
    "SegFormerHead",
    "ISAHead",
    "HVHead",
    "DomainDiscriminator",
    "CellPoseHead",
    "HV2Head",
    "PointNuHead",
    # "RPNHead",
    "TopdownHeatmapSimpleHead"
]
