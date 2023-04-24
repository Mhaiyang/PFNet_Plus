"""
 @Time    : 7/30/21 17:27
 @Author  : TaylorMei
 @Email   : haiyang.mei@outlook.com
 
 @Project : SSI2023_PFNet_Plus
 @File    : v33.py
 @Function:
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet

###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

###################################################################
# ############################ SimAM Module #######################
###################################################################
class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

###################################################################
# ##################### Context Enrichment Module #################
###################################################################
class Context_Enrichment_Module(nn.Module):
    def __init__(self, channel4, channel3, channel2, channel1):
        super(Context_Enrichment_Module, self).__init__()
        self.channel_gap = int(channel4 / 4)
        self.channel4 = channel4
        self.channel3 = channel3
        self.channel2 = channel2
        self.channel1 = channel1
        self.c4_w1 = nn.Parameter(torch.ones(1))
        self.c4_w2 = nn.Parameter(torch.ones(1))
        self.c4_w4 = nn.Parameter(torch.ones(1))
        self.c4_w8 = nn.Parameter(torch.ones(1))
        self.c3_w1 = nn.Parameter(torch.ones(1))
        self.c3_w2 = nn.Parameter(torch.ones(1))
        self.c3_w4 = nn.Parameter(torch.ones(1))
        self.c3_w8 = nn.Parameter(torch.ones(1))
        self.c2_w1 = nn.Parameter(torch.ones(1))
        self.c2_w2 = nn.Parameter(torch.ones(1))
        self.c2_w4 = nn.Parameter(torch.ones(1))
        self.c2_w8 = nn.Parameter(torch.ones(1))
        self.c1_w1 = nn.Parameter(torch.ones(1))
        self.c1_w2 = nn.Parameter(torch.ones(1))
        self.c1_w4 = nn.Parameter(torch.ones(1))
        self.c1_w8 = nn.Parameter(torch.ones(1))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                 nn.Conv2d(self.channel4, self.channel_gap, 1), nn.BatchNorm2d(self.channel_gap),
                                 nn.ReLU(),
                                 nn.Conv2d(self.channel_gap, self.channel4, 1), nn.BatchNorm2d(self.channel4),
                                 nn.ReLU())
        self.up43 = nn.Sequential(nn.Conv2d(self.channel4, self.channel3, 7, 1, 3), nn.BatchNorm2d(self.channel3),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up32 = nn.Sequential(nn.Conv2d(self.channel3, self.channel2, 7, 1, 3), nn.BatchNorm2d(self.channel2),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up21 = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3), nn.BatchNorm2d(self.channel1),
                                  nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.c4_1 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_1 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_1 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_1 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 1, dilation=1),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_2 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_2 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_2 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 2, dilation=2),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_4 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_4 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_4 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_4 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 4, dilation=4),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.c4_8 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel4), nn.ReLU())
        self.c3_8 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel3), nn.ReLU())
        self.c2_8 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.c1_8 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 8, dilation=8),
                                  nn.BatchNorm2d(self.channel1), nn.ReLU())
        self.cf4 = simam_module()
        self.cf3 = simam_module()
        self.cf2 = simam_module()
        self.cf1 = simam_module()
        self.output4 = nn.Sequential(nn.Conv2d(self.channel4, self.channel4, 3, 1, 1), nn.BatchNorm2d(self.channel4),
                                     nn.ReLU())
        self.output3 = nn.Sequential(nn.Conv2d(self.channel3, self.channel3, 3, 1, 1), nn.BatchNorm2d(self.channel3),
                                     nn.ReLU())
        self.output2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2),
                                     nn.ReLU())
        self.output1 = nn.Sequential(nn.Conv2d(self.channel1, self.channel1, 3, 1, 1), nn.BatchNorm2d(self.channel1),
                                     nn.ReLU())

    def forward(self, feature4, feature3, feature2, feature1):
        c4_input = feature4 + self.gap(feature4)
        c4_1 = self.c4_1(c4_input)
        c4_2 = self.c4_2(c4_input)
        c4_4 = self.c4_4(c4_input)
        c4_8 = self.c4_8(c4_input)
        c4 = self.c4_w1 * c4_1 + self.c4_w2 * c4_2 + self.c4_w4 * c4_4 + self.c4_w8 * c4_8
        c4 = self.cf4(c4)
        output4 = self.output4(feature4 + c4)

        c3_input = feature3 + self.up43(c4)
        c3_1 = self.c3_1(c3_input)
        c3_2 = self.c3_2(c3_input)
        c3_4 = self.c3_4(c3_input)
        c3_8 = self.c3_8(c3_input)
        c3 = self.c3_w1 * c3_1 + self.c3_w2 * c3_2 + self.c3_w4 * c3_4 + self.c3_w8 * c3_8
        c3 = self.cf3(c3)
        output3 = self.output3(feature3 + c3)

        c2_input = feature2 + self.up32(c3)
        c2_1 = self.c2_1(c2_input)
        c2_2 = self.c2_2(c2_input)
        c2_4 = self.c2_4(c2_input)
        c2_8 = self.c2_8(c2_input)
        c2 = self.c2_w1 * c2_1 + self.c2_w2 * c2_2 + self.c2_w4 * c2_4 + self.c2_w8 * c2_8
        c2 = self.cf2(c2)
        output2 = self.output2(feature2 + c2)

        c1_input = feature1 + self.up21(c2)
        c1_1 = self.c1_1(c1_input)
        c1_2 = self.c1_2(c1_input)
        c1_4 = self.c1_4(c1_input)
        c1_8 = self.c1_8(c1_input)
        c1 = self.c1_w1 * c1_1 + self.c1_w2 * c1_2 + self.c1_w4 * c1_4 + self.c1_w8 * c1_8
        c1 = self.cf1(c1)
        output1 = self.output1(feature1 + c1)

        return output4, output3, output2, output1

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        # self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        # map = self.map(sab)

        # return sab, map
        return sab

###################################################################
# ################ Pyramid Positioning Module #####################
###################################################################
class Pyramid_Positioning(nn.Module):
    def __init__(self, channel):
        super(Pyramid_Positioning, self).__init__()
        self.channel = channel
        self.channel_quarter = int(channel / 4)

        self.conv5 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((5, 5)))
        self.conv7 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((7, 7)))
        self.conv9 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                   nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((9, 9)))
        self.conv11 = nn.Sequential(nn.Conv2d(self.channel, self.channel_quarter, 3, 1, 1),
                                    nn.BatchNorm2d(self.channel_quarter), nn.ReLU(),
                                    nn.AdaptiveMaxPool2d((11, 11)))

        self.pm5 = Positioning(self.channel_quarter)
        self.pm7 = Positioning(self.channel_quarter)
        self.pm9 = Positioning(self.channel_quarter)
        self.pm11 = Positioning(self.channel_quarter)

        # self.up5 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up7 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up9 = nn.UpsamplingBilinear2d(size=(13, 13))
        # self.up11 = nn.UpsamplingBilinear2d(size=(13, 13))
        self.up5 = nn.UpsamplingBilinear2d(size=(15, 15))
        self.up7 = nn.UpsamplingBilinear2d(size=(15, 15))
        self.up9 = nn.UpsamplingBilinear2d(size=(15, 15))
        self.up11 = nn.UpsamplingBilinear2d(size=(15, 15))
        # self.up5 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up7 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up9 = nn.UpsamplingBilinear2d(size=(11, 11))
        # self.up11 = nn.UpsamplingBilinear2d(size=(11, 11))

        self.af = simam_module()
        self.fusion = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.BatchNorm2d(self.channel),
                                    nn.ReLU())
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)
        conv9 = self.conv9(x)
        conv11 = self.conv11(x)

        pm5 = self.pm5(conv5)
        pm7 = self.pm7(conv7)
        pm9 = self.pm9(conv9)
        pm11 = self.pm11(conv11)

        up5 = self.up5(pm5)
        up7 = self.up7(pm7)
        up9 = self.up9(pm9)
        up11 = self.up11(pm11)

        fusion = torch.cat([up5, up7, up9, up11], 1)
        fusion = self.fusion(fusion)
        fusion = fusion + x
        fusion = self.af(fusion)

        map = self.map(fusion)

        return fusion, map

###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map

###################################################################
# ########################## NETWORK ##############################
###################################################################
class V33(nn.Module):
    def __init__(self, backbone_path=None):
        super(V33, self).__init__()
        # params

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # context enrichment
        self.cem = Context_Enrichment_Module(512, 256, 128, 64)

        # positioning
        self.positioning = Pyramid_Positioning(512)

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # context enrichment
        ce4, ce3, ce2, ce1 = self.cem(cr4, cr3, cr2, cr1)

        # positioning
        positioning, predict4 = self.positioning(ce4)

        # focus
        focus3, predict3 = self.focus3(ce3, positioning, predict4)
        focus2, predict2 = self.focus2(ce2, focus3, predict3)
        focus1, predict1 = self.focus1(ce1, focus2, predict2)

        # rescale
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict1)
