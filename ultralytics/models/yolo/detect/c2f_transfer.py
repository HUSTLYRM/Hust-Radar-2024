import torch
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.extra_modules.prune_module import *

from ultralytics.nn.extra_modules.block import C2f_Faster, C2f_EMBC, RepNCSPELAN4

def transfer_weights_c2f_v2_to_c2f(c2f_v2, c2f):
    c2f.cv2 = c2f_v2.cv2
    c2f.m = c2f_v2.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    new_cv1 = Conv(c1=state_dict_v2['cv0.conv.weight'].size()[1],
                   c2=(state_dict_v2['cv0.conv.weight'].size()[0] + state_dict_v2['cv1.conv.weight'].size()[0]),
                   k=c2f_v2.cv1.conv.kernel_size,
                   s=c2f_v2.cv1.conv.stride)
    c2f.cv1 = new_cv1
    c2f.c1, c2f.c2 = state_dict_v2['cv0.conv.weight'].size()[0], state_dict_v2['cv1.conv.weight'].size()[0]
    state_dict['cv1.conv.weight'] = torch.cat([state_dict_v2['cv0.conv.weight'], state_dict_v2['cv1.conv.weight']], dim=0)

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        state_dict[f'cv1.bn.{bn_key}'] = torch.cat([state_dict_v2[f'cv0.bn.{bn_key}'], state_dict_v2[f'cv1.bn.{bn_key}']], dim=0)

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict[key] = state_dict_v2[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f_v2):
        attr_value = getattr(c2f_v2, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f, attr_name, attr_value)

    c2f.load_state_dict(state_dict)

def replace_c2f_v2_with_c2f(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f_v2):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f = C2f_infer(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights_c2f_v2_to_c2f(child_module, c2f)
            setattr(module, name, c2f)
        else:
            replace_c2f_v2_with_c2f(child_module)
    
    # for name, child_module in module.named_children():
    #     if isinstance(child_module, C2f_EMBC_v2):
    #         # Replace C2f with C2f_v2 while preserving its parameters
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f = C2f_EMBC_infer(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=1,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_v2_to_c2f(child_module, c2f)
    #         setattr(module, name, c2f)
    #     elif isinstance(child_module, C2f_v2):
    #         # Replace C2f with C2f_v2 while preserving its parameters
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f = C2f_infer(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=child_module.m[0].cv2.conv.groups,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_v2_to_c2f(child_module, c2f)
    #         setattr(module, name, c2f)
    #     else:
    #         replace_c2f_v2_with_c2f(child_module)

def infer_shortcut(bottleneck):
    try:
        c1 = bottleneck.cv1.conv.in_channels
        c2 = bottleneck.cv2.conv.out_channels
        return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add
    except:
        return False

def transfer_weights_c2f_to_c2f_v2(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def transfer_weights_elan_to_elan_v2(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.cv3 = c2f.cv3
    c2f_v2.cv4 = c2f.cv4

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c2f_with_c2f_v2(module):
    # for yolov8n.yaml
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights_c2f_to_c2f_v2(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)
    
    # for yolov8-Faster-GFPN-P2-EfficientHead.yaml
    # for name, child_module in module.named_children():
    #     if isinstance(child_module, C2f_Faster):
    #         # Replace C2f with C2f_v2 while preserving its parameters
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f_v2 = C2f_Faster_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=1,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_to_c2f_v2(child_module, c2f_v2)
    #         setattr(module, name, c2f_v2)
    #     elif isinstance(child_module, C2f):
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=child_module.m[0].cv2.conv.groups,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_to_c2f_v2(child_module, c2f_v2)
    #         setattr(module, name, c2f_v2)
    #     else:
    #         replace_c2f_with_c2f_v2(child_module)
    
    # for yolov8-BIFPN-EfficientRepHead.yaml
    # for name, child_module in module.named_children():
    #     if isinstance(child_module, C2f_EMBC):
    #         # Replace C2f with C2f_v2 while preserving its parameters
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f_v2 = C2f_EMBC_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=1,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_to_c2f_v2(child_module, c2f_v2)
    #         setattr(module, name, c2f_v2)
    #     elif isinstance(child_module, C2f):
    #         shortcut = infer_shortcut(child_module.m[0])
    #         c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
    #                         n=len(child_module.m), shortcut=shortcut,
    #                         g=child_module.m[0].cv2.conv.groups,
    #                         e=child_module.c / child_module.cv2.conv.out_channels)
    #         transfer_weights_c2f_to_c2f_v2(child_module, c2f_v2)
    #         setattr(module, name, c2f_v2)
    #     else:
    #         replace_c2f_with_c2f_v2(child_module)
    
    # for yolov8-repvit-RepNCSPELAN.yaml
    # for name, child_module in module.named_children():
    #     if isinstance(child_module, RepNCSPELAN4):
    #         # Replace C2f with C2f_v2 while preserving its parameters
    #         c2f_v2 = RepNCSPELAN4_v2(child_module.cv1.conv.in_channels, child_module.cv4.conv.out_channels,
    #                         child_module.cv1.conv.out_channels, child_module.cv3[-1].conv.out_channels, 1)
    #         transfer_weights_elan_to_elan_v2(child_module, c2f_v2)
    #         setattr(module, name, c2f_v2)
    #     else:
    #         replace_c2f_with_c2f_v2(child_module)