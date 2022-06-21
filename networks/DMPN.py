import torch
from torch import nn
import torch.nn.functional as F
import re

from .resnet import ResNet, Bottleneck
from .vgg import *
from .densenet import densenet161
from .deform_conv_v2 import DeformConv2d

k = 64

class DynamicWeights(nn.Module):

    def __init__(self, channels, rgb_inchannel, group=1, kernel=3, dilation=(1, 1, 1), shuffle=False, deform=None):
        super(DynamicWeights, self).__init__()
        in_channel = channels
        # down the channels of input image features
        self.smooth_rgb = nn.Sequential(nn.Conv2d(rgb_inchannel, in_channel, 1, bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.ReLU(inplace=True))
        # self.smooth_dep = nn.Sequential(nn.Conv2d(rgb_inchannel, in_channel, 1, bias=False),
        #                                 nn.BatchNorm2d(in_channel),
        #                                 nn.ReLU(inplace=True))

        self.filter1 = Warp(in_channel, groups_k=group)
        self.filter2 = Warp(in_channel, groups_k=group)
        self.filter3 = Warp(in_channel, groups_k=group)
        # self.filter4 = Warp(in_channel, groups_k=group)

        # inpalce


        if deform == 'deformatt':
            self.cata_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[0],
                                      dilation=dilation[0], bias=False)
            self.catb_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[1],
                                      dilation=dilation[1], bias=False)
            self.catc_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[2],
                                      dilation=dilation[2], bias=False)

            # learn kernel

            self.unfold1 = DeformConv2d(in_channel, in_channel, kernel_size=3, padding=dilation[0], stride=dilation[0])
            self.unfold2 = DeformConv2d(in_channel, in_channel, kernel_size=3, padding=dilation[1], stride=dilation[1])
            self.unfold3 = DeformConv2d(in_channel, in_channel, kernel_size=3, padding=dilation[2], stride=dilation[2])

            self.upsp1 = nn.Conv2d(in_channel, in_channel * 9, 1)
            self.upsp2 = nn.Conv2d(in_channel, in_channel * 9, 1)
            self.upsp3 = nn.Conv2d(in_channel, in_channel * 9, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 4, rgb_inchannel, 1, padding=0, bias=True),
                                    # group_norm(rgb_inchannel),
                                    nn.ReLU(inplace=True))

    def forward(self, rgb_feat, depth1, depth2, depth3):
        # blur_depth = x
        rgb_org = rgb_feat

        x = self.smooth_rgb(rgb_feat)
        N, C, H, W = x.size()
        R = C // self.group
        affinity1, filter_w1 = self.filter1(depth1)  # (B, 36, H, W), (B, 9, H, W)
        affinity2, filter_w2 = self.filter2(depth2)
        affinity3, filter_w3 = self.filter3(depth3)

        if self.deform == 'deformatt':
            offset_1 = self.cata_off(x)
            offset_2 = self.catb_off(x)
            offset_3 = self.catc_off(x)

            offset1 = offset_1[:, :18, :, :]
            filter_w1 = filter_w1.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w1 = filter_w1.sigmoid()

            offset2 = offset_2[:, :18, :, :]
            filter_w2 = filter_w2.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w2 = filter_w2.sigmoid()

            offset3 = offset_3[:, : 18, :, :]
            filter_w3 = filter_w3.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w3 = filter_w3.sigmoid()

        if self.deform == 'none':
            xd_unfold1 = self.unfold1(x)
            xd_unfold2 = self.unfold2(x)
            xd_unfold3 = self.unfold3(x)
        else:
            # print('x = ', x.size())
            xd_unfold1 = self.unfold1(x, offset1)
            xd_unfold2 = self.unfold2(x, offset2)
            xd_unfold3 = self.unfold3(x, offset3)

        if self.deform == 'deformatt':
            # print('xd_unfold1 = ', xd_unfold1.size())
            xd_unfold1 = self.upsp1(xd_unfold1)
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W)
            xd_unfold2 = self.upsp2(xd_unfold2)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W)
            xd_unfold3 = self.upsp3(xd_unfold3)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W)

            xd_unfold1 = xd_unfold1 * filter_w1
            xd_unfold2 = xd_unfold2 * filter_w2
            xd_unfold3 = xd_unfold3 * filter_w3

            xd_unfold1 = xd_unfold1.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        else:
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        ## N, K, H, W --> N*H*W
        # use softmax or not
        # affinity1 = affinity1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                             self.K)
        # affinity2 = affinity2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                             self.K)
        # affinity3 = affinity3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                            self.K)
        affinity1 = self.softmax(affinity1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                                                                                                                self.K)) # (BGHW, K)
        affinity2 = self.softmax(affinity2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                                                                                                                self.K)) # (BGHW, K)
        affinity3 = self.softmax(affinity3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                                                                                                                self.K)) # (BGHW, K)

        out1 = torch.bmm(xd_unfold1, affinity1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W) # (B, C, H, W)
        out2 = torch.bmm(xd_unfold2, affinity2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W) # (B, C, H, W)
        out3 = torch.bmm(xd_unfold3, affinity3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W) # (B, C, H, W)

        out = torch.cat((self.scale2(torch.cat((x, out1, out2, out3), 1)), rgb_org), 0)

        return out

class Warp(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1, groups_k=1, deform=False, normalize=None, att=False,
                 need_offset=False):
        super(Warp, self).__init__()
        self.group = groups
        self.need_offset = need_offset
        offset_groups = groups
        # if need_offset:
        self.off_conv = nn.Conv2d(channels, kernel * kernel * 2 * offset_groups + 9, 3,
                                  padding=dilation, dilation=dilation, bias=False)
        self.conv = DeformConv2d(channels, channels, kernel_size=3, padding=dilation, stride=dilation, bias=False,
                               )

        self.conv_1 = nn.Conv2d(channels, kernel * kernel * groups_k, kernel_size=3, padding=dilation,
                                dilation=dilation,
                                bias=False)
        self.bn = nn.BatchNorm2d(kernel * kernel * groups_k)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, H, W = x.size()

        R = C // self.group
        offset_filter = self.off_conv(x)
        offset = offset_filter[:, :18, :, :]
        filter = offset_filter[:, -9:, :, :]
        out = self.conv(x, offset)
        x = x + out
        x = self.relu(self.bn(self.conv_1(x)))
        return x, filter

class DMPN(nn.Module):
    def __init__(self, backbone):
        super(DMPN, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(1024, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2048, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

        # DMP部分
        dw_config = {}
        dw_group = dw_config.get('group', 4)
        dw_kernel = dw_config.get('kernel', 3)
        dw_dilation = dw_config.get('dilation', (1, 1, 1, 1))
        dw_shuffle = dw_config.get('shuffle', False)
        dw_deform = dw_config.get('deform', 'deformatt')

        self.dw_block2 = DynamicWeights(channels=256,
                                        rgb_inchannel=512,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)
        self.dw_block3 = DynamicWeights(channels=256,
                                        rgb_inchannel=1024,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)

        self.smooth_depth21 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth22 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth23 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.smooth_depth32 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth33 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth34 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self, x):
        # put tensor from Resnet backbone to compress model
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)

        feature_extract.append(self.CP[1](x[0]))
        feature_extract.append(self.CP[2](x[1]))

        depth1_layer2 = self.smooth_depth21(F.max_pool2d(x[1][1].unsqueeze(dim=0), 2))
        depth2_layer2 = self.smooth_depth22(x[2][1].unsqueeze(dim=0))
        depth3_layer2 = self.smooth_depth23(F.interpolate(x[3][1].unsqueeze(dim=0), scale_factor=2, mode='nearest'))
        dw2 = self.dw_block2(x[2][0].unsqueeze(dim=0), depth1_layer2, depth2_layer2, depth3_layer2)
        feature_extract.append(self.CP[3](dw2))

        depth2_layer3 = self.smooth_depth32(F.max_pool2d(x[2][1].unsqueeze(dim=0), 2))
        depth3_layer3 = self.smooth_depth33(x[3][1].unsqueeze(dim=0))
        depth4_layer3 = self.smooth_depth34(x[4][1].unsqueeze(dim=0))
        dw3 = self.dw_block3(x[3][0].unsqueeze(dim=0), depth2_layer3, depth3_layer3, depth4_layer3)
        feature_extract.append(self.CP[4](dw3))

        feature_extract.append(self.CP[5](x[4]))

        # for i in range(5):
        #     feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract  # list of tensor that compress model output

class DMPNVGG(nn.Module):
    def __init__(self, backbone):
        super(DMPNVGG, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

        # DMP部分
        dw_config = {}
        dw_group = dw_config.get('group', 4)
        dw_kernel = dw_config.get('kernel', 3)
        dw_dilation = dw_config.get('dilation', (1, 1, 1, 1))
        dw_shuffle = dw_config.get('shuffle', False)
        dw_deform = dw_config.get('deform', 'deformatt')

        self.dw_block2 = DynamicWeights(channels=256,
                                        rgb_inchannel=512,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)
        self.dw_block3 = DynamicWeights(channels=256,
                                        rgb_inchannel=512,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)

        self.smooth_depth21 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth22 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth23 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.smooth_depth32 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth33 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth34 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # for vgg16, the layers' names are different with the pretrained vgg16.pth, so the names should to be changed.
        renamed_dict = dict()
        for k, v in pretrained_dict.items():
            k = k.replace('features', 'layers')
            if k in model_dict:
                renamed_dict[k] = v
        # print('vgg16, pretrained:', pretrained_dict.items())
        # print('vgg16, renamed:', renamed_dict.items())
        model_dict.update(renamed_dict)
        # print('model, update:', model_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        # put tensor from VGG backbone to compress model
        feature_extract = []
        x = self.backbone(x)
        for i in range(3):
            feature_extract.append(self.CP[i](x[i]))

        depth1_layer2 = self.smooth_depth21(F.max_pool2d(x[2][1].unsqueeze(dim=0), 2))
        depth2_layer2 = self.smooth_depth22(x[3][1].unsqueeze(dim=0))
        depth3_layer2 = self.smooth_depth23(F.interpolate(x[4][1].unsqueeze(dim=0), scale_factor=2, mode='nearest'))
        dw2 = self.dw_block2(x[3][0].unsqueeze(dim=0), depth1_layer2, depth2_layer2, depth3_layer2)
        feature_extract.append(self.CP[3](dw2))

        depth2_layer3 = self.smooth_depth32(F.max_pool2d(x[3][1].unsqueeze(dim=0), 2))
        depth3_layer3 = self.smooth_depth33(x[4][1].unsqueeze(dim=0))
        depth4_layer3 = self.smooth_depth34(x[5][1].unsqueeze(dim=0))
        dw3 = self.dw_block3(x[4][0].unsqueeze(dim=0), depth2_layer3, depth3_layer3, depth4_layer3)
        feature_extract.append(self.CP[4](dw3))

        feature_extract.append(self.CP[5](x[5]))
        return feature_extract  # list of tensor that compress model output

class DMPNDensenet(nn.Module):
    def __init__(self, backbone):
        super(DMPNDensenet, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(96, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(384, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(768, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2112, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2208, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

        # DMP部分
        dw_config = {}
        dw_group = dw_config.get('group', 4)
        dw_kernel = dw_config.get('kernel', 3)
        dw_dilation = dw_config.get('dilation', (1, 1, 1, 1))
        dw_shuffle = dw_config.get('shuffle', False)
        dw_deform = dw_config.get('deform', 'deformatt')

        self.dw_block2 = DynamicWeights(channels=256,
                                        rgb_inchannel=768,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)
        self.dw_block3 = DynamicWeights(channels=256,
                                        rgb_inchannel=2112,
                                        group=dw_group,
                                        kernel=dw_kernel,
                                        dilation=dw_dilation,
                                        shuffle=dw_shuffle,
                                        deform=dw_deform)

        self.smooth_depth21 = nn.Sequential(nn.Conv2d(384, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth22 = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth23 = nn.Sequential(nn.Conv2d(2112, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.smooth_depth32 = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth33 = nn.Sequential(nn.Conv2d(2112, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth34 = nn.Sequential(nn.Conv2d(2208, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

    def load_pretrained_model(self, model_path):

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.backbone.load_state_dict(state_dict)

        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self, x):
        # put tensor from Resnet backbone to compress model
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)

        feature_extract.append(self.CP[1](x[0]))
        feature_extract.append(self.CP[2](x[1]))
        # feature_extract.append(self.CP[3](x[2]))

        depth1_layer2 = self.smooth_depth21(F.max_pool2d(x[1][1].unsqueeze(dim=0), 2))
        depth2_layer2 = self.smooth_depth22(x[2][1].unsqueeze(dim=0))
        depth3_layer2 = self.smooth_depth23(F.interpolate(x[3][1].unsqueeze(dim=0), scale_factor=2, mode='nearest'))
        dw2 = self.dw_block2(x[2][0].unsqueeze(dim=0), depth1_layer2, depth2_layer2, depth3_layer2)
        feature_extract.append(self.CP[3](dw2))
        #
        depth2_layer3 = self.smooth_depth32(F.max_pool2d(x[2][1].unsqueeze(dim=0), 2))
        depth3_layer3 = self.smooth_depth33(x[3][1].unsqueeze(dim=0))
        depth4_layer3 = self.smooth_depth34(x[4][1].unsqueeze(dim=0))
        dw3 = self.dw_block3(x[3][0].unsqueeze(dim=0), depth2_layer3, depth3_layer3, depth4_layer3)
        feature_extract.append(self.CP[4](dw3))

        feature_extract.append(self.CP[5](x[4]))
        #
        # for i in range(5):
        #     feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract  # list of tensor that compress model output

class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            part1 = list_x[i][0].unsqueeze(dim=0)
            part2 = list_x[i][1].unsqueeze(dim=0)
            sum = (part1 + part2 + (part1 * part2))
            resl.append(sum)
        return resl

class FAModule(nn.Module):
    def __init__(self):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu)
        self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k / 2), 1), self.relu,
                                          nn.Conv2d(int(k / 2), int(k / 4), 3, 1, 1), self.relu)
        self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu,
                                          nn.Conv2d(int(k / 4), int(k / 4), 5, 1, 2), self.relu)
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k / 4), 1), self.relu)

    def forward(self, x_cm, x_fa):
        # element-wise addition
        x = x_cm

        for i in range(len(x_fa)):
            x += x_fa[i]
        # aggregation
        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2(x)
        x_branch3 = self.conv_branch3(x)
        x_branch4 = self.conv_branch4(x)

        x = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)
        return x


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class DMPNet(nn.Module):
    def __init__(self, base_model_cfg, DMPN, cm_layers, feature_aggregation_module, JL_score_layers,
                 DCF_score_layers, upsampling):
        super(DMPNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.DMPN = DMPN
        self.FA = nn.ModuleList(feature_aggregation_module)
        self.upsampling = nn.ModuleList(nn.ModuleList(upsampling[i]) for i in range(0, 4))
        self.score_JL = JL_score_layers
        self.score_DCF = DCF_score_layers
        self.cm = cm_layers

    def forward(self, x):
        x = self.DMPN(x)
        x_cm = self.cm(x)
        s_coarse = self.score_JL(x[5])
        x_cm = x_cm[::-1]
        x_fa = []
        x_fa_temp = []
        x_fa.append(self.FA[4](x_cm[1], x_cm[0]))
        x_fa_temp.append(x_fa[0])
        for i in range(len(x_cm) - 2):
            for j in range(len(x_fa)):
                x_fa_temp[j] = self.upsampling[i][i - j](x_fa[j])
            x_fa.append(self.FA[3 - i](x_cm[i + 2], x_fa_temp))
            x_fa_temp.append(x_fa[-1])

        s_final = self.score_DCF(x_fa[-1])
        return s_final, s_coarse


def build_model(network='resnet101', base_model_cfg='resnet'):
    feature_aggregation_module = []
    for i in range(5):
        feature_aggregation_module.append(FAModule())
    upsampling = []
    for i in range(0, 4):
        upsampling.append([])
        for j in range(0, i + 1):
            upsampling[i].append(
                nn.ConvTranspose2d(k, k, kernel_size=2 ** (j + 2), stride=2 ** (j + 1), padding=2 ** (j)))
    if base_model_cfg == 'resnet':
        parameter = [3, 4, 23, 3] if network == 'resnet101' else [3, 4, 6, 3]
        backbone = ResNet(Bottleneck, parameter)
        return DMPNet(base_model_cfg, DMPN(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
    elif base_model_cfg == 'vgg':
        backbone = vgg(network=network)
        return DMPNet(base_model_cfg, DMPNVGG(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
    elif base_model_cfg == 'densenet':
        backbone = densenet161()
        return DMPNet(base_model_cfg, DMPNDensenet(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                  ScoreLayer(k), upsampling)
