import torch
import torch_pruning as tp
from typing import Sequence

from .rep_block import DiverseBranchBlock
class DiverseBranchBlockPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: DiverseBranchBlock, idxs: Sequence[int]):
        # prune for dbb_origin
        tp.prune_conv_out_channels(layer.dbb_origin.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.dbb_origin.bn, idxs)
        
        # prune for dbb_avg and dbb_1x1
        if hasattr(layer.dbb_avg, 'conv'):
            # dbb_avg
            tp.prune_conv_out_channels(layer.dbb_avg.conv, idxs)
            tp.prune_batchnorm_out_channels(layer.dbb_avg.bn.bn, idxs)
            
            # dbb_1x1
            tp.prune_conv_out_channels(layer.dbb_1x1.conv, idxs)
            tp.prune_batchnorm_out_channels(layer.dbb_1x1.bn, idxs)
        
        tp.prune_batchnorm_out_channels(layer.dbb_avg.avgbn, idxs)
        
        # prune for dbb_1x1_kxk
        tp.prune_conv_out_channels(layer.dbb_1x1_kxk.conv2, idxs)
        tp.prune_batchnorm_out_channels(layer.dbb_1x1_kxk.bn2, idxs)
        
        # update out_channels
        layer.out_channels = layer.out_channels - len(idxs)
        return layer
            
    def prune_in_channels(self, layer: DiverseBranchBlock, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        
        # prune for dbb_origin
        tp.prune_conv_in_channels(layer.dbb_origin.conv, idxs)
        
        # prune for dbb_avg and dbb_1x1
        if hasattr(layer.dbb_avg, 'conv'):
            # dbb_avg
            tp.prune_conv_in_channels(layer.dbb_avg.conv, idxs)
            
            # dbb_1x1
            tp.prune_conv_in_channels(layer.dbb_1x1.conv, idxs)
        
        # prune for dbb_1x1_kxk
        if hasattr(layer.dbb_1x1_kxk, 'idconv1'):
            layer.dbb_1x1_kxk.idconv1.id_tensor = self._prune_parameter_and_grad(layer.dbb_1x1_kxk.idconv1.id_tensor, keep_idxs=keep_idxs, pruning_dim=1)
            tp.prune_conv_in_channels(layer.dbb_1x1_kxk.idconv1.conv, idxs)
        elif hasattr(layer.dbb_1x1_kxk, 'conv1'):
            tp.prune_conv_in_channels(layer.dbb_1x1_kxk.conv1, idxs)
        
        # update in_channels
        layer.in_channels = layer.in_channels - len(idxs)
        return layer
        
    def get_out_channels(self, layer: DiverseBranchBlock):
        return layer.out_channels
    
    def get_in_channels(self, layer: DiverseBranchBlock):
        return layer.in_channels
    
    def get_channel_groups(self, layer: DiverseBranchBlock):
        return layer.groups

from ..backbone.convnextv2 import LayerNorm
class LayerNormPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer:LayerNorm, idxs: Sequence[int]):
        num_features = layer.normalized_shape[0]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        keep_idxs.sort()
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, -1)
        layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, -1)
        layer.normalized_shape = (len(keep_idxs),)
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return layer.normalized_shape[0]

    def get_in_channels(self, layer):
        return layer.normalized_shape[0]

from ..modules.conv import RepConv
class RepConvPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: RepConv, idxs: Sequence[int]):
        layer.c2 = layer.c2 - len(idxs)
        
        tp.prune_conv_out_channels(layer.conv1.conv, idxs)
        tp.prune_conv_out_channels(layer.conv2.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.conv1.bn, idxs)
        tp.prune_batchnorm_out_channels(layer.conv2.bn, idxs)
        return layer
    
    def prune_in_channels(self, layer: RepConv, idxs: Sequence[int]):
        layer.c1 = layer.c1 - len(idxs)
        
        tp.prune_conv_in_channels(layer.conv1.conv, idxs)
        tp.prune_conv_in_channels(layer.conv2.conv, idxs)
        return layer

    def get_out_channels(self, layer: RepConv):
        return layer.c2

    def get_in_channels(self, layer: RepConv):
        return layer.c1

    def get_in_channel_groups(self, layer: RepConv):
        return layer.g
    
    def get_out_channel_groups(self, layer: RepConv):
        return layer.g

from ..extra_modules.dyhead_prune import DyHeadBlock_Prune
class DyHeadBlockPruner(tp.pruner.BasePruningFunc):
    def prune_in_channels(self, layer: DyHeadBlock_Prune, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        
        layer.spatial_conv_low.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_low.conv.weight, keep_idxs, 1)
        layer.spatial_conv_mid.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_mid.conv.weight, keep_idxs, 1)
        layer.spatial_conv_high.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_high.conv.weight, keep_idxs, 1)
        tp.prune_conv_in_channels(layer.spatial_conv_offset, idxs)
        
        return layer
    
    def prune_out_channels(self, layer: DyHeadBlock_Prune, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.spatial_conv_low.conv.weight.size(0))) - set(idxs))
        keep_idxs.sort()
        keep_idxs = keep_idxs[:len(keep_idxs) - (len(keep_idxs) % self.get_out_channel_groups(layer))]
        idxs = list(set(range(layer.spatial_conv_low.conv.weight.size(0))) - set(keep_idxs))
        
        # spatial_conv
        layer.spatial_conv_low.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_low.conv.weight, keep_idxs, 0)
        layer.spatial_conv_mid.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_mid.conv.weight, keep_idxs, 0)
        layer.spatial_conv_high.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_high.conv.weight, keep_idxs, 0)
        layer.spatial_conv_low.norm = self.prune_groupnorm(layer.spatial_conv_low.norm, keep_idxs)
        layer.spatial_conv_mid.norm = self.prune_groupnorm(layer.spatial_conv_mid.norm, keep_idxs)
        layer.spatial_conv_high.norm = self.prune_groupnorm(layer.spatial_conv_high.norm, keep_idxs)
        
        # scale_attn_module
        tp.prune_conv_in_channels(layer.scale_attn_module[1], idxs)
        
        # task_attn_module
        dim = layer.task_attn_module.oup
        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]
        tp.prune_linear_in_channels(layer.task_attn_module.fc[0], idxs)
        tp.prune_linear_out_channels(layer.task_attn_module.fc[2], idxs_repeated)
        layer.task_attn_module.oup = layer.task_attn_module.oup - len(idxs)
        
        return layer
    
    def get_out_channels(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_low.conv.weight.size(0)

    def get_in_channels(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_low.conv.weight.size(1)
    
    def get_in_channel_groups(self, layer: DyHeadBlock_Prune):
        return 1
    
    def get_out_channel_groups(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_high.norm.num_groups
    
    def prune_groupnorm(self, layer: torch.nn.GroupNorm, keep_idxs):
        layer.num_channels = keep_idxs
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

from .block import RepConvN
class RepConvNPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: RepConvN, idxs: Sequence[int]):
        layer.c2 = layer.c2 - len(idxs)
        
        tp.prune_conv_out_channels(layer.conv1.conv, idxs)
        tp.prune_conv_out_channels(layer.conv2.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.conv1.bn, idxs)
        tp.prune_batchnorm_out_channels(layer.conv2.bn, idxs)
        return layer
    
    def prune_in_channels(self, layer: RepConvN, idxs: Sequence[int]):
        layer.c1 = layer.c1 - len(idxs)
        
        tp.prune_conv_in_channels(layer.conv1.conv, idxs)
        tp.prune_conv_in_channels(layer.conv2.conv, idxs)
        return layer

    def get_out_channels(self, layer: RepConvN):
        return layer.c2

    def get_in_channels(self, layer: RepConvN):
        return layer.c1

    def get_in_channel_groups(self, layer: RepConvN):
        return layer.g
    
    def get_out_channel_groups(self, layer: RepConvN):
        return layer.g