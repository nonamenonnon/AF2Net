import torch
import torch.nn as nn
import torch.nn.functional as F
from smt import smt_s

from thop import profile

from torch.nn import init
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MSP(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MSP, self).__init__()
        # branch1

        self.branch1 = nn.Sequential(
            BasicConv2d(inplanes, inplanes, kernel_size=1, stride=1),
            BasicConv2d(inplanes, (inplanes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv2d((inplanes // 2) * 3, 2 * inplanes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv2d(2 * inplanes, inplanes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = BasicConv2d(3 * inplanes, inplanes, 3, padding=1)
        self.upsample2 = upsample
        self.stride2 = stride


    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out2 = self.conv4(out2)
        out2 = self.bn4(out2)
        out3 = self.branch1(x)
        # out3 = self.branch2(out3)
        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2, out3), 1))
        out += residual
        out = self.relu(out)

        return out

class AFF(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AFF, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv5 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.conv5(x)
        z = self.bn1(z)
        # z = self.relu(z)
        # z = self.conv5(z)
        # z = self.bn1(z)
        z = self.sigmoid(z)
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y * z

class AF2Net(nn.Module):
    def __init__(self, channel=64):
        super(AF2Net, self).__init__()

        self.smt = smt_s()

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.Translayer3_1 = BasicConv2d(256, 64, 1)
        self.Translayer4_1 = BasicConv2d(512, 64, 1)

        self.AFF = AFF(in_channel=64, ratio=4).to('cuda')
        self.MSP = MSP(64, 64)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self,x):

        rgb_list = self.smt(x)

        r4 = rgb_list[3]
        r3 = rgb_list[2]
        r2 = rgb_list[1]
        r1 = rgb_list[0]

        r2 = self.Translayer2_1(r2)
        r3 = self.Translayer3_1(r3)
        r4 = self.Translayer4_1(r4)

        r4_ = self.up1(r4)

        r34 = self.AFF(r4_+r3)
        r34_ = self.MSP(r34)
        r34 = self.up1(r34_)

        r23 = self.AFF(r34+r2)
        r23_ = self.MSP(r23)
        r23 = self.up1(r23_)

        r1234 = self.AFF(r23+r1)
        r1234 = self.MSP(r1234)

        r123 = F.interpolate(self.predtrans1(r1234), size=416, mode='bilinear')
        r12 = F.interpolate(self.predtrans2(r23), size=416, mode='bilinear')
        r1 = F.interpolate(self.predtrans3(r34), size=416, mode='bilinear')

        return r123, r12, r1

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    flops, params = profile(AF2Net(x), (x,))
    print('flops: %.4f G, parms: %.4f M' % (flops / 1000000000.0, params / 1000000.0))
