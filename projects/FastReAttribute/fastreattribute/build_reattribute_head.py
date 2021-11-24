
from fastreid.modeling.heads import REID_HEADS_REGISTRY
import torch

def build_reattribute_heads(cfg, **kwargs):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    reattribute_heads_cfg_nodes = cfg.MODEL.ATTRIBUTE_HEADS
    reattribute_heads = [REID_HEADS_REGISTRY.get(cfg_node['NAME'])(cfg_node, **kwargs) for cfg_node in reattribute_heads_cfg_nodes]
    
    return reattribute_heads

def build_reattribute_option_heads(cfg, **kwargs):
    """
    Build OPTION_HEADS defined by `cfg.MODEL.OPTION_HEADS.NAME`.
    """
    reattribute_option_heads_cfg_nodes = cfg.MODEL.OPTION_HEADS
    reattribute_option_heads = [REID_HEADS_REGISTRY.get(cfg_node['NAME'])(cfg_node, **kwargs) for cfg_node in reattribute_option_heads_cfg_nodes]
    
    return reattribute_option_heads