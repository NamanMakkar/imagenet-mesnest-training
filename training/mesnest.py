import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        grad_hx = i.sigmoid()

        grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 
        
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class SplitAttn(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=Mish, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        c2 = c2 or c1
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = c2 * radix
        if rd_channels is None:
            attn_chs = _make_divisible(c1 * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        #self.conv = nn.Conv2d(
        #   c1, mid_chs, kernel_size, stride, padding, dilation,
        #    groups=groups * radix, bias=bias, **kwargs)
        self.conv = GhostConv(c1,mid_chs,k=kernel_size,s=stride,g=groups*radix)
        #self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        #self.act0 = act_layer()
        #self.fc1 = nn.Conv2d(c2, attn_chs, 1, groups=groups)
        self.fc1 = GhostConv(c2, attn_chs, 1, 1)
        #self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        #self.act1 = act_layer()
        #self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.fc2 = GhostConv(attn_chs, mid_chs, 1, 1, act=False)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        #x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        #x_gap = self.bn1(x_gap)
        #x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        output = out.contiguous()
        return output

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                Mish(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = Mish()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class MSBlock(nn.Module):
    def __init__(self, c1, c2, stride, use_se, radix=2, expand_ratio=1):
        super(MSBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(c1 * expand_ratio)
        self.identity = stride == 1 and c1 == c2
        self.stride = stride
        if not self.identity:
            if stride > 1:
                self.down = nn.Sequential(nn.MaxPool2d(3,2,1),
                                          nn.Conv2d(c1,c2,1,1,0,bias=False))
            else:
                self.down = nn.Conv2d(c1,c2,1,1,0,bias=False)
            
        
        if use_se:
            self.conv = nn.Sequential(
                #pw
                GhostConv(c1, hidden_dim, 1, 1),
                #mixed conv
                MixConv2d(hidden_dim,hidden_dim,k=(3,5,7,9),s=stride,equal_ch=True),
                #se
                SELayer(c1, hidden_dim),
                #linear
                GhostConv(hidden_dim, c2, 1, 1, act=False)
            )
        else:
            self.conv = nn.Sequential(
                #mixconv
                MixConv2d(c1,hidden_dim,k=(3,5,7,9,11,13),s=1,equal_ch=True),
                #split attention - radix = 2 - SK-Unit,
                SplitAttn(c1=hidden_dim,c2=c2,stride=stride,groups=1,radix=radix,
                          rd_ratio=0.25, act_layer=Mish, norm_layer=nn.BatchNorm2d),
            )

    def forward(self, x):
        if self.identity:
            out = x + self.conv(x)
        else:
            out = self.down(x) + self.conv(x)
        return channel_shuffle(out,2)

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        Mish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        Mish()
    )

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class MESNeSt(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(MESNeSt, self).__init__()
        self.cfgs = cfgs
        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MSBlock
        for t, c, n, s, use_se, radix in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, use_se, radix, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

def mesnest_s(**kwargs):
    """
    Constructs a Mixconv-Efficient-Shuffle-NeSt-S model
    """
    cfgs = [
        # t, c, n, s, SE, radix
        [1,  24,  2, 1, 0, 2],
        [4,  48,  4, 2, 0, 2],
        [4,  64,  4, 2, 0, 2],
        [4, 128,  6, 2, 1, 1],
        [6, 160,  9, 1, 1, 1],
        [6, 256, 15, 2, 1, 1],
    ]
    return MESNeSt(cfgs, **kwargs)


def mesnest_m(**kwargs):
    """
    Constructs a Mixconv-Efficient-Shuffle-NeSt-M model
    """
    cfgs = [
        # t, c, n, s, SE, radix
        [1,  24,  3, 1, 0, 2],
        [4,  48,  5, 2, 0, 2],
        [4,  80,  5, 2, 0, 2],
        [4, 160,  7, 2, 1, 1],
        [6, 176, 14, 1, 1, 1],
        [6, 304, 18, 2, 1, 1],
        [6, 512,  5, 1, 1, 1],
    ]
    return MESNeSt(cfgs, **kwargs)


def mesnest_l(**kwargs):
    """
    Constructs a Mixconv-Efficient-Shuffle-NeSt-L model
    """
    cfgs = [
        # t, c, n, s, SE, radix
        [1,  32,  4, 1, 0, 2],
        [4,  64,  7, 2, 0, 2],
        [4,  96,  7, 2, 0, 2],
        [4, 192, 10, 2, 1, 1],
        [6, 224, 19, 1, 1, 1],
        [6, 384, 25, 2, 1, 1],
        [6, 640,  7, 1, 1, 1],
    ]
    return MESNeSt(cfgs, **kwargs)


def mesnest_xl(**kwargs):
    """
    Constructs a Mixconv-Efficient-Shuffle-NeSt-XL model
    """
    cfgs = [
        # t, c, n, s, SE, radix
        [1,  32,  4, 1, 0, 2],
        [4,  64,  8, 2, 0, 2],
        [4,  96,  8, 2, 0, 2],
        [4, 192, 16, 2, 1, 1],
        [6, 256, 24, 1, 1, 1],
        [6, 512, 32, 2, 1, 1],
        [6, 640,  8, 1, 1, 1],
    ]
    return MESNeSt(cfgs, **kwargs)