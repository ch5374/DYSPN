import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import scipy.special as sc
from .pmath import *


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, nonlin=None, bias=False, dim=-1):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)  # torch.Size([8, 64])
        self.dim = dim
        self.nonlin = nonlin
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(self.weight.device)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = mobius_matvec(self.weight, x, c=c)

        if self.nonlin is not None:
            mv = mobius_fn_apply(self.nonlin, mv, c=self.c)

        if self.bias is None:
            return project(mv, c=c)
        else:
            bias = expmap0(self.bias, c=c)
            return project(mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(self.in_features, self.out_features,
                                                                       self.bias is not None, self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c=0.1, train_c=False, train_x=False, ball_dim=None, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:

            #self.c = nn.Parameter(torch.Tensor([c, ]))
            #### 원래 curvature
            self.cu = nn.Parameter(torch.rand((228 * 304, 1))).cuda()
            #### 픽셀마다 다른 커버쳐
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = RiemannianGradient
        self.riemannian.c = c
        self.relu = nn.ReLU()
        #### 추가

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        self.c = self.relu(self.cu)
        #### 양수가 나와야됨

        if self.train_x:
            xp = project(expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(project(expmap(xp, x, c=self.c), c=self.c))

        return self.grad_fix(project(expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c=0.1, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c, ]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = project(expmap0(self.xp, c=self.c), c=self.c)
            return self.pmath.logmap(xp, x, c=self.c)
        return logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)


class GeoConv(nn.Module):
    def __init__(self, in_feat, out_feat, c, kernel=3, strides=1, padding=0, dilation=1, nonlin=None, train_c=False,
                 mode='naive'):
        super().__init__()
        assert mode in ['naive', 'model1', 'model1_only_weight', 'model1_only_sort', 'model1+']
        print('Geo-Conv Mode Set To: ', mode)

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.K = kernel
        self.F = 5  # window(receptive field)
        self.S = strides
        self.P = padding
        self.D = dilation
        self.non_lin = lambda x: nonlin(x) if nonlin else x

        self.type = mode
        self.c = c
        self.e2p = ToPoincare(self.c, train_c=train_c)
        self.p2e = FromPoincare(self.c, train_c=train_c)
        self.scale = torch.tensor(sc.beta(in_feat * kernel * kernel / 2, 0.5) / sc.beta(in_feat / 2, 0.5))
        self.hyplinear = HypLinear(in_feat * kernel * kernel, out_feat, self.c, nonlin=None, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Shape of Output Convolution
        B, C, H, W = x.shape

        if self.type in ['model1_only_weight', 'model1_only_sort', 'model1', 'model1+']:
            # Unfold HW with K [NCHW -> N(C*K*K)(H*W) -> NNW(K*K)C]
            x_unfold = F.unfold(x, kernel_size=self.K, dilation=self.D, padding=self.P, stride=self.S)
            tm = x_unfold.view(B, C, self.K * self.K, -1).permute(0, 3, 2, 1).reshape(B, H, W, self.K * self.K, C)
            # Geodesic distance
            if self.type == 'model1+':
                tm_eu = x_unfold.view(B, C, self.K * self.K, -1).permute(0, 3, 2, 1).reshape(B, H, W, -1)
            tm_center = x.permute(0, 2, 3, 1).unsqueeze(dim=3)
            geo_dist = dist(x=self.e2p(tm_center), y=self.e2p(tm), c=self.c)

            if self.type in ['model1_only_sort', 'model1', 'model1+']:
                # Sort with geodesic distance
                geo_dist, geo_dist_idx = torch.sort(geo_dist, dim=3)
                tm = torch.gather(tm, dim=3, index=geo_dist_idx.unsqueeze(4).expand(B, H, W, self.K * self.K, C))

            # Weighted aggregation
            if self.type in ['model1_only_weight', 'model1', 'model1+']:
                tm = tm / self.softmax(geo_dist).unsqueeze(dim=4)  # Optional
            # Concat [NNW(K*K)C -> NNW(K*K*C)]
            tm = tm.reshape(B, H, W, -1)

        else:
            # Unfold HW with K [NCHW -> N(C*K*K)(H*W) -> NNW(F*F*C)]
            x_unfold = F.unfold(x, kernel_size=self.K, dilation=self.D, padding=self.P, stride=self.S)
            tm = x_unfold.view(B, C, self.K * self.K, -1).permute(0, 3, 2, 1).reshape(B, H, W, -1)

        if self.type == 'model1+':
            tm = (tm + tm_eu) / 2

        vector = tm * self.scale  # Beta concatenation
        out = self.p2e(self.hyplinear(self.e2p(vector)))  # Hyperbolic Linear
        out = self.non_lin(out).permute(0, 3, 1, 2)  # Permute [NHWC -> NCHW]
        return out
