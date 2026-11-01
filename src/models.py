"""Small-image ResNet-18 and controlled DW/SE variants, trained from scratch.

Architecture references: He et al., CVPR 2016; Howard et al., arXiv:1704.04861;
Hu et al., CVPR 2018. Implemented here without downloading pretrained weights.
"""
import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(),
                                nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        scale = self.fc(self.pool(x).flatten(1))[:, :, None, None]
        return x * scale


def conv(cin, cout, stride, depthwise):
    if depthwise:
        # The intermediate DW/PW boundary introduces no extra BN/activation.
        return nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                             nn.Conv2d(cin, cout, 1, bias=False))
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, depthwise, attention):
        super().__init__()
        self.branch = nn.Sequential(conv(cin, cout, stride, depthwise), nn.BatchNorm2d(cout),
                                    nn.ReLU(), conv(cout, cout, 1, depthwise),
                                    nn.BatchNorm2d(cout), SE(cout) if attention else nn.Identity())
        self.shortcut = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


class TrafficResNet(nn.Module):
    def __init__(self, variant='A', classes=43):
        super().__init__()
        if variant not in 'ABCDE' or len(variant) != 1:
            raise ValueError('Variant must be A, B, C, D or E')
        self.variant = variant
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.stages = nn.ModuleList()
        cin = 64
        for i, cout in enumerate([64, 128, 256, 512]):
            dw = variant in ('B', 'D') or (variant == 'E' and i >= 2)
            se = variant in ('C', 'D')
            self.stages.append(nn.Sequential(BasicBlock(cin, cout, 1 if i == 0 else 2, dw, se),
                                             BasicBlock(cout, cout, 1, dw, se)))
            cin = cout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        return self.fc(self.pool(self.forward_features(x)).flatten(1))


def complexity(model, size=48):
    """Count Conv2d/Linear MACs only, excluding bias adds and elementwise ops."""
    counts, handles = {}, []
    def hook(name):
        def count(layer, inputs, out):
            factor = (layer.in_channels // layer.groups * layer.kernel_size[0] * layer.kernel_size[1]
                      if isinstance(layer, nn.Conv2d) else layer.in_features)
            counts[name] = out.numel() * factor
        return count
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook(name)))
    device = next(model.parameters()).device
    mode = model.training
    model.eval()
    with torch.inference_mode():
        shape = list(model(torch.zeros(1, 3, size, size, device=device)).shape)
    for handle in handles:
        handle.remove()
    model.train(mode)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                conv_linear_macs=sum(counts.values()), input_shape=[1, 3, size, size], output_shape=shape,
                stage_macs=[sum(v for k, v in counts.items() if k.startswith(f'stages.{i}.')) for i in range(4)],
                scope='Conv2d/Linear MACs only; 1 MAC = 1 multiply-accumulate')
