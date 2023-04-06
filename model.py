import torch.nn as nn
import torch
import math


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),

            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


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
        out = self.convVCR2(Xc)
        Zc = self.bn2(out)
        # ----------------CS模块----------------------
        Mc = []
        out = self.CSgp(Zc)
        out = self.convCSq(out)
        query = self.csbnq(out)
        query = query.view(query.size(0), query.size(1), query.size(2) * query.size(3))
        query = torch.transpose(query, dim1=1, dim0=2)
        query = torch.chunk(query, c, dim=2)
        query = list(query)
        out = self.convCSk(Zc)
        key = self.csbnk(out)
        key = key.view(key.size(0), key.size(1), key.size(2) * key.size(3))
        key = torch.chunk(key, c, dim=1)
        key = list(key)
        Zc_ = Zc.view(Zc.size(0), Zc.size(1), Zc.size(2) * Zc.size(3))
        Zc_ = torch.chunk(Zc_, c, dim=1)
        Zc_ = list(Zc_)
        for i in range(len(query)):
            in_softmax = torch.matmul(query[i], key[i])
            out_softmax = self.cssoftmax(in_softmax)
            Mc.append(out_softmax)
            out_softmax = torch.transpose(out_softmax, dim0=1, dim1=2)
            Zc_[i] = torch.matmul(Zc_[i], out_softmax)
        Zc_out = torch.cat(Zc_, dim=1)
        conv_in = Zc_out.view(Zc_out.size(0), Zc_out.size(1),
                              int(math.sqrt(Zc_out.size(2))), int(math.sqrt(Zc_out.size(2))))
        out = self.convCSout(conv_in)
        hc = self.csbnout(out)
        # ----------------CR模块----------------------
        out = self.convCR(hc)
        out = self.crbn(out)
        A = self.CRtanh(out)
        A = A.view(A.size(0), A.size(1), A.size(2)*A.size(3))
        A = torch.chunk(A, c, dim=1)
        A = torch.cat(A, dim=2)
        H = hc.view(hc.size(0), hc.size(1), hc.size(2)*hc.size(3))
        H = torch.chunk(H, c, dim=1)
        H = torch.cat(H, dim=2)
        H = torch.transpose(H, dim0=1, dim1=2)
        out = torch.matmul(A, H)
        out = torch.add(out, H)
        out = torch.chunk(out, c, dim=1)
        out = torch.cat(out, dim=2)
        out = torch.transpose(out, dim0=1, dim1=2)
        out = out.view(out.size(0), out.size(1), out.size(2), out.size(2))
        out = self.CRbn1(out)
        hc_ = self.CRrelu(out)
        hc_ = hc_.view(hc_.size(0), hc_.size(1), out.size(2) * out.size(3))
        # ----------------CM模块----------------------
        Mchc = []
        Mc = torch.cat(Mc, dim=1)
        Mc_ = torch.div(Mc, torch.max(Mc))

        Mc_ = torch.chunk(Mc_, c, dim=1)
        hc_ = torch.chunk(hc_, c, dim=1)
        Mc_ = list(Mc_)
        hc_ = list(hc_)
        for i in range(len(Mc_)):
            out = torch.matmul(hc_[i], Mc_[i])
            Mchc.append(out)
        Mchc = torch.cat(Mchc, dim=1)
        Mchc = Mchc.view(Mchc.size(0), Mchc.size(1), int(math.sqrt(Mchc.size(2))), int(math.sqrt(Mchc.size(2))))

        alpha = self.convCMa(Mchc)
        beta = self.convCMb(Mchc)

        out = torch.mul(Zc, alpha)
        out = torch.add(out, beta)
        Xc_ = self.CMrelu(out)
        cm_out = torch.mul(Zc, Xc_)
        out = self.convVCR3(cm_out)
        OUT = self.bn3(out)

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

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

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


        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
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

        self.w = nn.Parameter(torch.Tensor([0.4, 0.6]), requires_grad=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.cbam = CBAMLayer(2048)          # 输入channel应与实际输入深度匹配
        # self.conv_cbam = nn.Conv2d(256, 2048, groups=groups, kernel_size=8, stride=8)
        self.vcr = VCRNet()
        # self.conv_up = nn.Conv2d(4096, 2048, groups=groups, kernel_size=1, stride=1)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            # self.fc = nn.Linear(1024 * block.expansion, num_classes)      # 对应concat融合
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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        cbam = self.cbam.forward(x)
        # cbam = self.conv_cbam(cbam)     # 上采样
        # # ————————————2—————————————
        x = self.vcr.forward(x)
        # x = torch.cat((x, cbam), 1)            # 1) concat融合
        # x = self.conv_up(x)
        # x = torch.add(cbam, x)                 # 2)特征相加
        x = cbam * self.w[0] + x * self.w[1]     # 3)特征加权
        # x = self.conv_up(x)
        # # ————————————3—————————————
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resinet(num_classes=1000, include_top=True):
    # resinet: Residual Semantic Reinforcement Network
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)



