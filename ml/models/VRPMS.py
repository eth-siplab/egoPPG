import os
import torch.nn as nn
import torch.optim as optim
from source.ml.models.base_networks import *
import torch.nn.functional as F
import torch
import numpy as np

from source.ml.models.resnet import resnet10 as Encoder


class LFA(nn.Module):
    def __init__(self, base_filter,video_length, in_channels):
        super(LFA, self).__init__()

        self.conv1=ConvBlock3D(in_channels, base_filter, 3, 1,1, activation='lrelu', norm=None)

        res1= [
            ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        self.res1= nn.Sequential(*res1)

        self.downsample_1=nn.Upsample(scale_factor=(1,0.5,0.5))
        self.downsample_2=nn.Upsample(scale_factor=(1,0.25,0.25))

        self.upsample_1 = nn.Upsample(scale_factor=(1,2,2))
        self.upsample_2 = nn.Upsample(scale_factor=(1,4,4))

        self.mod_gap=nn.AdaptiveAvgPool3d((video_length,1,1))
        self.mod_conv1d_1=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_1=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_1=nn.LSTM(2, 1, num_layers=1, batch_first=True)
        # , bidirectional = True,
        self.mod_conv1d_2=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_2=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_2=nn.LSTM(2, 1, num_layers=1, batch_first=True)

        self.mod_conv1d_3=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_3=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_3=nn.LSTM(2, 1, num_layers=1, batch_first=True )

        self.finalconv1=ConvBlock3D(base_filter*3, in_channels, 1, 1,padding=0, norm=None)


    def forward(self, input, ratio):


        B, C, T, H, W = input.size()
        # input = input.transpose(1,2)
        feat = self.conv1(input)
        feat=self.res1(feat)
        featdown1=self.downsample_1(feat)
        featdown2 =self.downsample_2(feat)

        mod1feat=self.mod_gap(feat)
        mod1feat=mod1feat.squeeze(3)
        mod1feat=mod1feat.squeeze(3)

        mod1feat=self.mod_conv1d_1(mod1feat)
        ratio= ratio.unsqueeze(1)

        fuse1=torch.cat((mod1feat, ratio), 1)

        mod1feat = self.mod_res1d_1(fuse1)
        mod1feat=mod1feat.transpose(1,2)
        mod1feat = self.mod_lstm_1(mod1feat)[0]
        mod1feat=mod1feat.view(B,1,T,1,1)
        aftermod_1=torch.mul(feat,mod1feat)

        mod2feat=self.mod_gap(featdown1)
        mod2feat=mod2feat.squeeze(3)
        mod2feat=mod2feat.squeeze(3)
        mod2feat=self.mod_conv1d_2(mod2feat)
        fuse2=torch.cat((mod2feat, ratio), 1)
        mod2feat = self.mod_res1d_2(fuse2)
        mod2feat=mod2feat.transpose(1,2)
        mod2feat = self.mod_lstm_2(mod2feat)[0]
        mod2feat=mod2feat.view(B,1,T,1,1)
        aftermod_2=torch.mul(featdown1,mod2feat)
        aftermod_2=self.upsample_1(aftermod_2)

        mod3feat=self.mod_gap(featdown2)
        mod3feat=mod3feat.squeeze(3)
        mod3feat=mod3feat.squeeze(3)
        mod3feat=self.mod_conv1d_3(mod3feat)
        fuse3=torch.cat((mod3feat, ratio), 1)
        mod3feat = self.mod_res1d_3(fuse3)
        mod3feat=mod3feat.transpose(1,2)
        mod3feat = self.mod_lstm_3(mod3feat)[0]
        mod3feat=mod3feat.view(B,1,T,1,1)
        aftermod_3=torch.mul(featdown2,mod3feat)
        aftermod_3=self.upsample_2(aftermod_3)

        finalfeat=torch.cat((aftermod_1, aftermod_2,aftermod_3), 1)
        finalfeat=self.finalconv1(finalfeat)

        return finalfeat


class Expert(nn.Module):
    def __init__(self, base_filter, video_length, w, h):
        super(Expert, self).__init__()
        self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.ra = RABlock(base_filter, w, h, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.res1(feat)
        feat = self.ra(feat)
        feat = self.res2(feat)
        feat = self.gap(feat)
        feat = feat.squeeze(3)
        feat = feat.squeeze(3)
        feat = self.cov1d(feat)
        return feat

class Gating(nn.Module):
    def __init__(self, base_filter, video_length, num_expert):
        super(Gating, self).__init__()
        self.conv1=ConvBlock3D(base_filter, base_filter, 3, 1,1, activation='relu', norm=None)
        self.conv2=ConvBlock3D(base_filter, base_filter, 3, 1,1, activation='relu', norm=None)

        self.gap=nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d=nn.Conv1d(in_channels=base_filter, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.act = nn.Softmax(dim=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat=self.conv2(feat)
        feat=self.gap(feat)
        feat=feat.squeeze(3)
        feat=feat.squeeze(3)
        feat=self.cov1d(feat)
        feat = self.act(feat)
        return feat

class REA(nn.Module):
    def __init__(self, base_filter, video_length, num_expert, device, in_channels, w, h):
        super(REA, self).__init__()
        self.num_expert = num_expert
        self.experts = nn.ModuleList([Expert(base_filter,video_length, w, h) for _ in range(self.num_expert)])
        self.gating = Gating(base_filter, video_length, num_expert)
        self.encoder = Encoder()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.device = device


    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):

        feat = self.encoder(input)
        gates = self.gating(feat)
        B, C, T, H, W = input.size()

        index1=index2=[i for i in range(int(self.num_expert**0.5))]
        count=0
        init_rppg=torch.zeros((B,T), device=self.device)
        for m in index1:
            for n in index2:
                cutfeat = feat[:, :, :, int(H/(self.num_expert**0.5)*m):int(H/(self.num_expert**0.5)*(m+1)),
                               int(W/(self.num_expert**0.5)*n):int(W/(self.num_expert**0.5)*(n+1))]
                cutfeat = self.experts[count](cutfeat).squeeze(1)
                cut_rppg=torch.mul(cutfeat, gates[:, count, :])
                init_rppg+=cut_rppg
                count+=1

        init_rppg=init_rppg.unsqueeze(1)
        rppg_fuse = self.conv1d(init_rppg)
        return rppg_fuse


class VRPMS(nn.Module):
    def __init__(self, base_filter, num_negative, video_length, num_expert, device, w, h, in_channels=1):
        super(VRPMS, self).__init__()

        self.LFA=LFA(base_filter, video_length, in_channels)
        self.REA=REA(base_filter, video_length,num_expert, device, in_channels, w, h)
        self.num_negative=num_negative


    def forward(self, input, positive1, positive2, neighbor1, neighbor2, neighbor3, ratio_array):
        neg_rppgarr=[]
        negative_arr=[]
        for i in range(self.num_negative):
            negative = self.LFA(input,ratio_array[:,i])
            neg_rppg = self.REA(negative).squeeze(1)
            neg_rppgarr.append(neg_rppg)
            negative_arr.append(negative)

        pos_rppg1 = self.REA(positive1).squeeze(1)
        pos_rppg2 = self.REA(positive2).squeeze(1)
        neighbor_rppg1 = self.REA(neighbor1).squeeze(1)
        neighbor_rppg2 = self.REA(neighbor2).squeeze(1)
        neighbor_rppg3 = self.REA(neighbor3).squeeze(1)

        negative_arr = [negative for negative in negative_arr]

        return neg_rppgarr,pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3,negative_arr