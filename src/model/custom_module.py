import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''import torch
gks3 = 7  # guide kernel size
pad7 = []
pad5 = []
pad3 = []
pad1 = []
for i in range(gks3):
    for j in range(gks3):
        top = i
        bottom = gks3 - 1 - i
        left = j
        right = gks3 - 1 - j
        if top == 6 or bottom == 6 or left == 6 or right == 6:
          pad7.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
        elif top == 5 or bottom == 5 or left == 5 or right == 5:
          pad5.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
        elif top == 4 or bottom == 4 or left == 4 or right == 4:
          pad3.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
        else:
          pad1.append(torch.nn.ZeroPad2d((left, right, top, bottom)))'''

gks = 5
pad = [i for i in range(gks * gks)]
shift = torch.zeros(gks * gks, 4)
for i in range(gks):
    for j in range(gks):
        top = i
        bottom = gks - 1 - i
        left = j
        right = gks - 1 - j
        pad[i * gks + j] = torch.nn.ZeroPad2d((left, right, top, bottom))
        # shift[i*gks + j, :] = torch.tensor([left, right, top, bottom])
mid_pad = torch.nn.ZeroPad2d(((gks - 1) / 2, (gks - 1) / 2, (gks - 1) / 2, (gks - 1) / 2))
zero_pad = pad[0]

gks2 = 3  # guide kernel size
pad2 = [i for i in range(gks2 * gks2)]
shift = torch.zeros(gks2 * gks2, 4)
for i in range(gks2):
    for j in range(gks2):
        top = i
        bottom = gks2 - 1 - i
        left = j
        right = gks2 - 1 - j
        pad2[i * gks2 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

gks3 = 7  # guide kernel size
pad3 = [i for i in range(gks3 * gks3)]
shift = torch.zeros(gks3 * gks3, 4)
for i in range(gks3):
    for j in range(gks3):
        top = i
        bottom = gks3 - 1 - i
        left = j
        right = gks3 - 1 - j
        pad3[i * gks3 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels)
    )

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)



class CSPNGenerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        # normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        # padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if (self.kernel_size == 3):
                zero_pad = pad2[t]
            elif (self.kernel_size == 5):
                zero_pad = pad[t]
            elif (self.kernel_size == 7):
                zero_pad = pad3[t]
            if (t < int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad[t] = zero_pad(guide[:, t:t + 1, :, :])
            elif (t > int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad[t] = zero_pad(guide[:, t - 1:t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1)
        return guide_weight

class CSPNGenerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        # normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        # padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if (self.kernel_size == 3):
                zero_pad = pad2[t]
            elif (self.kernel_size == 5):
                zero_pad = pad[t]
            elif (self.kernel_size == 7):
                zero_pad = pad3[t]
            if (t < int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad[t] = zero_pad(guide[:, t:t + 1, :, :])
            elif (t > int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad[t] = zero_pad(guide[:, t - 1:t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1)
        return guide_weight

class CSPN(nn.Module):
    def __init__(self, kernel_size):
        super(CSPN, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, guide_weight, hn, h0):

        # CSPN
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if (self.kernel_size == 3):
                zero_pad = pad2[t]
            elif (self.kernel_size == 5):
                zero_pad = pad[t]
            elif (self.kernel_size == 7):
                zero_pad = pad3[t]
            if (t == half):
                result_pad[t] = zero_pad(h0)
            else:
                result_pad[t] = zero_pad(hn)
        guide_result = torch.cat([result_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1)
        # guide_result = torch.cat([result0_pad, result1_pad, result2_pad, result3_pad,result4_pad, result5_pad, result6_pad, result7_pad, result8_pad], 1)

        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[:, int((self.kernel_size - 1) / 2):-int((self.kernel_size - 1) / 2),
                       int((self.kernel_size - 1) / 2):-int((self.kernel_size - 1) / 2)]

        return guide_result.unsqueeze(dim=1)


class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        guide = self.generate(feature)

        # normalization in standard CSPN
        # '''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        # '''
        # weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output = torch.cat((half1, guide_mid, half2), dim=1)
        return output


class CSPNGenerateAccelerate_softmax(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate_softmax, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size, kernel_size=3, stride=1, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, feature):
        guide = self.generate(feature)
        guide = self.softplus(guide)

        guide_sum = torch.sum(guide.abs(), dim=1, keepdim=True)
        output = torch.div(guide, guide_sum)

        return output

class CSPNGenerateAccelerate_softmax_dyspn(nn.Module):
    def __init__(self, in_channels, kernel_size, hyperbolic=False):
        super(CSPNGenerateAccelerate_softmax_dyspn, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size, kernel_size=3, stride=1, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, feature):
        guide = self.generate(feature)
        guide = self.softplus(guide)

        return guide


class CSPNGenerateAccelerate_softmax_dyspn_normal(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate_softmax_dyspn_normal, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size + 1, kernel_size=3, stride=1, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, feature):
        guide = self.generate(feature)
        guide = self.softplus(guide)
        guide_sum = torch.sum(guide.abs(), dim=1, keepdim=True)
        output = torch.div(guide, guide_sum)

        return output

def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size - 1) / 2))
    return kernel


class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, kernel, input, input0):  # with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        if self.kernel_size == 3:
          input_im2col = F.unfold(input, self.kernel_size, 1, 1, 1)
        elif self.kernel_size == 7:
          input_im2col = F.unfold(input, self.kernel_size, 1, 3, 1)

        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index:mid_index + 1, :] = input0

        output = (input_im2col * kernel).sum(dim=1)
        return output.view(bs, 1, h, w)

class DYSPNAccelerate(nn.Module):
    def __init__(self, kernel_size):
        super(DYSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, kernel, input, input0, attention, i):
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        if self.kernel_size == 3:
          input_im2col = F.unfold(input, self.kernel_size, 1, 1, 1)
        elif self.kernel_size == 7:
          input_im2col = F.unfold(input, self.kernel_size, 1, 3, 1)

        new_kernel = torch.zeros_like(kernel)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        line7 = [j for j in range(self.kernel_size * self.kernel_size) if 0 <= j <= 7 or 13 <= j <= 14 or 20 <= j <= 21 or 27 <= j <= 28 or 34 <= j <= 35 or 41 <= j <= 48]
        line5 = [j for j in range(self.kernel_size * self.kernel_size) if 8 <= j <= 12 or j == 15 or j == 19 or j == 22 or j == 26 or j == 29 or j == 33 or 36 <= j <= 40]
        line3 = [j for j in range(self.kernel_size * self.kernel_size) if 16 <= j <= 18 or j == 23 or j == 25 or 30 <= j <= 32]

        new_kernel[:, line7, :] = \
            kernel[:, line7, :] * attention[:, i, 3].unsqueeze(1)

        new_kernel[:, line5, :] = \
            kernel[:, line5, :] * attention[:, i, 2].unsqueeze(1)

        new_kernel[:, line3, :] = \
            kernel[:, line3, :] * attention[:, i, 1].unsqueeze(1)

        new_kernel[:,  mid_index, :] = \
            kernel[:,  mid_index, :] * attention[:, i, 0]

        new_kernel_sum = torch.sum(new_kernel.abs(), dim=1, keepdim=True)
        new_kernel = torch.div(new_kernel, new_kernel_sum)
        new_kernel = new_kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)

        #DS
        #input_h = input_im2col[:, mid_index:mid_index + 1, :]
        input_im2col[:, mid_index:mid_index + 1, :] = input0
        #input_im2col_ = torch.cat([input_im2col, input_h], dim=1)

        output = (input_im2col * new_kernel).sum(dim=1)
        return output.view(bs, 1, h, w)

class DYSPNAccelerate_just_ex(nn.Module):
    def __init__(self, kernel_size):
        super(DYSPNAccelerate_just_ex, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, kernel, input, input0, attention, i):
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        if self.kernel_size == 3:
          input_im2col = F.unfold(input, self.kernel_size, 1, 1, 1)
        elif self.kernel_size == 7:
          input_im2col = F.unfold(input, self.kernel_size, 1, 3, 1)

        new_kernel = torch.zeros_like(kernel)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        line7 = [j for j in range(self.kernel_size * self.kernel_size) if 0 <= j <= 7 or 13 <= j <= 14 or 20 <= j <= 21 or 27 <= j <= 28 or 34 <= j <= 35 or 41 <= j <= 48]#24
        line5 = [j for j in range(self.kernel_size * self.kernel_size) if 8 <= j <= 12 or j == 15 or j == 19 or j == 22 or j == 26 or j == 29 or j == 33 or 36 <= j <= 40]#16
        line3 = [j for j in range(self.kernel_size * self.kernel_size) if 16 <= j <= 18 or j == 23 or j == 25 or 30 <= j <= 32] #8

        new_kernel[:, line7, :] = \
            kernel[:, line7, :] * attention[:, i, 3].unsqueeze(1)

        new_kernel[:, line5, :] = \
            kernel[:, line5, :] * attention[:, i, 2].unsqueeze(1)

        new_kernel[:, line3, :] = \
            kernel[:, line3, :] * attention[:, i, 1].unsqueeze(1)

        new_kernel[:,  mid_index, :] = \
            kernel[:,  mid_index, :] * attention[:, i, 0]

        new_kernel[:, self.kernel_size * self.kernel_size, :] = \
            kernel[:, self.kernel_size * self.kernel_size, :]

        new_kernel_sum = torch.sum(new_kernel.abs(), dim=1, keepdim=True)
        new_kernel = torch.div(new_kernel, new_kernel_sum)
        new_kernel = new_kernel.reshape(bs, self.kernel_size * self.kernel_size + 1, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        inputh0 = input_im2col[:, mid_index:mid_index + 1,:]
        input_im2col[:, mid_index:mid_index + 1, :] = input0
        input_im2col_ = torch.cat([input_im2col, inputh0], dim=1)

        output = (input_im2col_ * new_kernel).sum(dim=1)
        return output.view(bs, 1, h, w)

class CSPNAccelerate_dyspn_normal(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate_dyspn_normal, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0, attention, i):
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        if self.kernel_size == 3:
          input_im2col = F.unfold(input, self.kernel_size, 1, 1, 1)
        elif self.kernel_size == 7:
          input_im2col = F.unfold(input, self.kernel_size, 1, 3, 1)

        new_kernel = torch.zeros_like(kernel)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)

        line7 = [j for j in range(self.kernel_size * self.kernel_size) if 0 <= j <= 7 or 13 <= j <= 14 or 20 <= j <= 21 or 27 <= j <= 28 or 34 <= j <= 35 or 41 <= j]
        line5 = [j for j in range(self.kernel_size * self.kernel_size) if 8 <= j <= 12 or j == 15 or j == 19 or j == 22 or j == 26 or j == 29 or j == 33 or 36 <= j <= 40]
        line3 = [j for j in range(self.kernel_size * self.kernel_size) if 16 <= j <= 18 or j == 23 or j == 25 or 29 <= j <= 31]

        new_kernel[:, line7, :] = \
            kernel[:, line7, :] * attention[:, i, 0].unsqueeze(1)

        new_kernel[:, line5, :] = \
            kernel[:, line5, :] * attention[:, i, 1].unsqueeze(1)

        new_kernel[:, line3, :] = \
            kernel[:, line3, :] * attention[:, i, 2].unsqueeze(1)

        new_kernel[:, mid_index, :] = \
            kernel[:, mid_index, :] * attention[:, i, 3]

        new_kernel[:, self.kernel_size * self.kernel_size, :] = \
            kernel[:, self.kernel_size * self.kernel_size, :]

        new_kernel_sum = torch.sum(kernel.abs(), dim=1, keepdim=True)
        new_kernel = torch.div(new_kernel, new_kernel_sum)

        new_kernel = new_kernel.reshape(bs, self.kernel_size * self.kernel_size + 1, h * w)

        input0 = input0.view(bs, 1, h * w)
        #input_im2col[:, mid_index:mid_index + 1, :] = input0
        input_im2col = torch.cat([input_im2col, input0], dim=1)

        output = (input_im2col * new_kernel).sum(dim=1)
        return output.view(bs, 1, h, w)





