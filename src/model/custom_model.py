from .common import *
import torch
import torch.nn as nn
from .custom_module import CSPNAccelerate, DYSPNAccelerate_just_ex

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        if self.args.model_name == 'S2D':
            self.num_neighbors = 0
        elif self.args.model_name == 'CSPN':
            self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel # 49
        elif self.args.model_name == 'DYSPN':
            self.num_neighbors = self.args.prop_kernel * self.args.prop_kernel + 1# 50


        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)

        if self.args.network == 'resnet18':
            net = get_resnet18(not self.args.from_scratch)
        elif self.args.network == 'resnet34':
            net = get_resnet34(not self.args.from_scratch)
        else:
            raise NotImplementedError

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/8
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/4
        self.dec4 = convt_bn_relu(256+512, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Guidance Branch
        # 1/1
        if self.args.model_name != 'S2D':
            self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                        padding=1)
            self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                        padding=1, bn=False, relu=False)
            self.softplus = nn.Softplus()

        if self.args.model_name == 'DYSPN':
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

            self.att1_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.att0_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 24, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        if self.args.model_name == 'CSPN':
            self.prop_layer = CSPNAccelerate(self.args.prop_kernel)

        elif self.args.model_name == 'DYSPN':
            self.prop_layer = DYSPNAccelerate_just_ex(self.args.prop_kernel)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        if self.args.model_name != 'S2D':
            gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
            guide_ = self.gd_dec0(self._concat(gd_fd1, fe1))
            #guide_ = self.softplus(guide_)
            if self.args.guide_normal:
                guide_sum = torch.sum(guide_.abs(), dim=1, keepdim=True)
                guide = torch.div(guide_, guide_sum)
            else:
                guide = guide_

        if self.args.model_name == 'DYSPN':
            # Confidence Decoding
            # Attention Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))

            att_fd1 = self.att1_dec1(self._concat(fd2, fe2))
            attention = self.att0_dec0(self._concat(att_fd1, fe1))
            # print(attention.shape)
            # Only n-times of num_gpus supprot
            bs, _, h, w = attention.shape
            if bs == 1:
                attention = attention.view(bs, 6, 4,
                                       self.args.patch_height, self.args.patch_width)
            elif bs > 1:
                attention = attention.view(int(self.args.batch_size / self.args.num_gpus), 6, 4,
                                           self.args.patch_height, self.args.patch_width)
            else:
                raise Exception('Batch size must be multiple of number of gpus, bs={}, gpus={}'.format(bs, self.args.num_gpus))
            if self.args.attention_normal:
                attention_sum = torch.sum(attention, dim=2, keepdim=True)
                attention = torch.div(attention, attention_sum)

        if self.args.model_name == 'CSPN':
            sparse_mask = dep.sign()
            depth = pred_init
            for _ in range(self.args.prop_time):
                depth = dep * sparse_mask + (1 - sparse_mask) * depth
                depth = self.prop_layer(guide, depth, pred_init)
            depth = torch.clamp(depth, min=0)

        elif self.args.model_name == 'DYSPN':
            sparse_mask = dep.sign()
            depth = pred_init
            sparse_depth = dep * confidence
            if not self.args.sparse_order_reverse:
                for i in range(self.args.prop_time):  # dyspn prop_time default 6
                    depth = sparse_mask * sparse_depth + (1 - sparse_mask) * depth
                    depth = self.prop_layer(guide, depth, pred_init, attention, i)
            else:
                for i in range(self.args.prop_time):  # dyspn reverse order
                    depth = self.prop_layer(guide, depth, pred_init, attention, i)
                    depth = sparse_mask * sparse_depth + (1 - sparse_mask) * depth

            depth = torch.clamp(depth, min=0)


        if self.args.model_name == 'S2D':
            pred_init = torch.clamp(pred_init, min=0)
            p = None
            guide = None
            confidence = None
            attention = None
            output = {'pred': pred_init, 'pred_init': p,
                      'guidance': guide, 'confidence': confidence, 'attention': attention}

        elif self.args.model_name == 'CSPN':
            confidence = None
            attention = None
            output = {'pred': depth, 'pred_init': pred_init,
                  'guidance': guide, 'confidence': confidence, 'attention': attention}

        elif self.args.model_name == 'DYSPN':
            output = {'pred': depth, 'pred_init': pred_init,
                  'guidance': guide, 'confidence': confidence, 'attention': attention}

        return output
