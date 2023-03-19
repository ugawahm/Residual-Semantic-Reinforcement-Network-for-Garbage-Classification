import torch.nn as nn
import torch
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class VCRNet(nn.Module):

    def __init__(self, d=2048, c=32, p=64, p_=4):
        super(VCRNet, self).__init__()
        self.convVCR1 = nn.Conv2d(in_channels=d, out_channels=c*p, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(c*p)
        self.convVCR2 = nn.Conv2d(in_channels=c*p, out_channels=c*p, groups=c, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        self.bn2 = nn.BatchNorm2d(c*p)

        # ----------------CS模块----------------------
        self.CSgp = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.convCSq = nn.Conv2d(in_channels=c*p, out_channels=c*p_, groups=c, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.csbnq = nn.BatchNorm2d(c*p_)

        self.convCSk = nn.Conv2d(in_channels=c * p, out_channels=c * p_, groups=c, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.csbnk = nn.BatchNorm2d(c * p_)

        self.cssoftmax = nn.Softmax2d()

        self.convCSout = nn.Conv2d(in_channels=c * p, out_channels=c * p_, groups=c, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.csbnout = nn.BatchNorm2d(c * p_)
        # ----------------CR模块----------------------
        self.convCR = nn.Conv2d(in_channels=c * p_, out_channels=c*c, groups=c, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.crbn = nn.BatchNorm2d(c * c)

        self.CRtanh = nn.Tanh()

        self.CRbn1 = nn.BatchNorm2d(c*p_)

        self.CRrelu = nn.ReLU()
        # ----------------CM模块----------------------
        self.convCMa = nn.Conv2d(in_channels=c * p_, out_channels=c * p, groups=c, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.cmbna = nn.BatchNorm2d(c * p)

        self.convCMb = nn.Conv2d(in_channels=c * p_, out_channels=c * p, groups=c, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.cmbnb = nn.BatchNorm2d(c * p)

        self.CMrelu = nn.ReLU()

        self.convVCR3 = nn.Conv2d(in_channels=c*p, out_channels=d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(d)

    def forward(self, x, c=32):
        out = self.convVCR1(x)
        Xc = self.bn1(out)
        # print('Xc = ', Xc.size())

        out = self.convVCR2(Xc)
        Zc = self.bn2(out)
        # print('Zc = ', Zc.size())         #  torch.Size([16, 2048, 7, 7])

        # ----------------CS模块----------------------
        Mc = []
        out = self.CSgp(Zc)
        out = self.convCSq(out)
        query = self.csbnq(out)
        query = query.view(query.size(0), query.size(1), query.size(2) * query.size(3))
        query = torch.transpose(query, dim1=1, dim0=2)
        query = torch.chunk(query, c, dim=2)
        query = list(query)
        # print('query = ', query[0].size())          # torch.Size([16, 1, 4])
        out = self.convCSk(Zc)
        key = self.csbnk(out)
        key = key.view(key.size(0), key.size(1), key.size(2) * key.size(3))
        # print('key = ', key.size())                 # torch.Size([16, 128, 49])
        key = torch.chunk(key, c, dim=1)
        key = list(key)
        Zc_ = Zc.view(Zc.size(0), Zc.size(1), Zc.size(2) * Zc.size(3))    # torch.Size([16, 2048, 49])
        Zc_ = torch.chunk(Zc_, c, dim=1)
        Zc_ = list(Zc_)
        for i in range(len(query)):
            in_softmax = torch.matmul(query[i], key[i])
            out_softmax = self.cssoftmax(in_softmax)
            # softmax_list.append(out_softmax)
            Mc.append(out_softmax)     # torch.Size([16, 1, 49])
            out_softmax = torch.transpose(out_softmax, dim0=1, dim1=2)    # torch.Size([16, 49, 1])
            # print(Zc_[i].size())
            Zc_[i] = torch.matmul(Zc_[i], out_softmax)
        # print('Zc_ = ', Zc_[0].size())
        Zc_out = torch.cat(Zc_, dim=1)
        # print('Zc_out = ', Zc_out.size())
        conv_in = Zc_out.view(Zc_out.size(0), Zc_out.size(1),
                              int(math.sqrt(Zc_out.size(2))), int(math.sqrt(Zc_out.size(2))))  # [16, 2048, 1, 1]
        # print('conv_in = ', conv_in.size())
        out = self.convCSout(conv_in)
        hc = self.csbnout(out)          # [16, 128, 1, 1]
        # print('hc = ', hc.size())
        # ----------------CR模块----------------------
        out = self.convCR(hc)
        out = self.crbn(out)
        A = self.CRtanh(out)       # [16, 1024, 1, 1]
        A = A.view(A.size(0), A.size(1), A.size(2)*A.size(3))
        A = torch.chunk(A, c, dim=1)
        A = torch.cat(A, dim=2)
        # print('A', A.size())       # [16, 32, 32]
        H = hc.view(hc.size(0), hc.size(1), hc.size(2)*hc.size(3))
        H = torch.chunk(H, c, dim=1)  # [16, 4, 1]
        H = torch.cat(H, dim=2)
        # print('H:', H.size())     # [16, 4, 32]
        H = torch.transpose(H, dim0=1, dim1=2)
        # print('H1:', H.size())
        out = torch.matmul(A, H)
        # print('out1=', out.size())      # [16, 32, 4]
        out = torch.add(out, H)
        # print('out2=', out.size())
        out = torch.chunk(out, c, dim=1)
        # print('out,,', out[1].size())
        out = torch.cat(out, dim=2)
        out = torch.transpose(out, dim0=1, dim1=2)
        out = out.view(out.size(0), out.size(1), out.size(2), out.size(2))
        out = self.CRbn1(out)
        hc_ = self.CRrelu(out)
        hc_ = hc_.view(hc_.size(0), hc_.size(1), out.size(2) * out.size(3))
        # print('hc_ = ', hc_.size())

        # ----------------CM模块----------------------
        Mchc = []
        Mc = torch.cat(Mc, dim=1)    # [16, 32, 49]
        Mc_ = torch.div(Mc, torch.max(Mc))
        # print('mc', Mc_.size())
        Mc_ = torch.chunk(Mc_, c, dim=1)
        hc_ = torch.chunk(hc_, c, dim=1)
        Mc_ = list(Mc_)
        hc_ = list(hc_)
        for i in range(len(Mc_)):
            out = torch.matmul(hc_[i], Mc_[i])
            Mchc.append(out)
        # print('mchc', Mchc[1].size())       # [16, 4, 49]
        Mchc = torch.cat(Mchc, dim=1)
        Mchc = Mchc.view(Mchc.size(0), Mchc.size(1), int(math.sqrt(Mchc.size(2))), int(math.sqrt(Mchc.size(2))))
        # print('mchc', Mchc.size())        # [16, 128, 7, 7]

        alpha = self.convCMa(Mchc)
        # print('alpha = ',alpha.size())
        beta = self.convCMb(Mchc)
        # print('beta = ',beta.size())

        out = torch.mul(Zc, alpha)
        out = torch.add(out, beta)
        Xc_ = self.CMrelu(out)
        # print('Xc_ = ',Xc_.size())

        cm_out = torch.mul(Zc, Xc_)
        # print('cm_out = ',cm_out.size())

        out = self.convVCR3(cm_out)
        OUT = self.bn3(out)
        # print('OUT = ',OUT.size())

        result = torch.add(OUT, x)

        return result


class BasicBlock(nn.Module):   #残差结构定义（18/34层）
    expansion = 1              #采用的卷积和个数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 加入注意力机制
        # self.ca = ChannelAttention(out_channel)
        # self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 加入注意力机制
        # self.ca = ChannelAttention(out_channel * self.expansion)
        # self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,                # 不同残差结构: basciblock (18/34层) 和 blockneck (50层以上)
                 blocks_num,           # 残差结构数目: 如34层为 [3,4,6,3]
                 num_classes=1000,     # 训练集分类个数
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(self.in_channel)
        # self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # self.ca1 = ChannelAttention(self.in_channel)
        # self.sa1 = SpatialAttention()

        self.vcr = VCRNet()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # # ————————————1—————————————
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # # ————————————2—————————————
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = self.vcr.forward(x)
        # # ————————————3—————————————
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
