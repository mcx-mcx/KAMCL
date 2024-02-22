# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torchvision.models.resnet import resnet18,resnet50,resnet101
import torch.nn.functional as F
import math
from layers import seq2vec
from torch.autograd import Variable
from layers.gpo import GPO

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class  ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.pool_2x2 = nn.MaxPool2d(4)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.gpo = GPO(32,32)
        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)
  

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # Lower Feature
        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        # Higher Feature
        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)

        feature = f4.view(f4.shape[0], 512, -1)#2048
        feature = feature.permute(0,2,1)
  
        length=[64.]*feature.shape[0]#64
        length=torch.tensor(length).cuda()  
        solo_feature = self.gpo(feature,length)[0]
 
        return lower_feature, higher_feature, solo_feature


class Skipthoughts_Embedding_Module(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.mlp=nn.Sequential(nn.Linear(512, 512),nn.ReLU(),nn.Linear(512, 512),nn.ReLU())

        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])
        self.gpo = GPO(32,32)
        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text):
        x_t_vec = self.seq2vec(input_text,lengths=None)
        x_t_vec = self.to_out(x_t_vec)
        x_t_vec_tem = self.mlp(x_t_vec)
        tem3 = torch.cat([x_t_vec,x_t_vec_tem],dim=1)
        length=[float(tem3.size(1))]
        length=torch.tensor(length).cuda() 

        x_t_vec_all = self.gpo(tem3,length)[0]
        out = F.tanh(x_t_vec_all)
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12

class low_module(nn.Module):
    def __init__(self):
        super(low_module, self).__init__()
        self.gpo=GPO(32,32)
        self.fc = nn.Linear(4096,512)
    def forward(self, lower_feature):
        lower_feature = lower_feature.view(lower_feature.shape[0],lower_feature.shape[1],-1)
        length=[float(lower_feature.shape[1])]*lower_feature.shape[0]
        length=torch.tensor(length).cuda() 
        lower= self.gpo(lower_feature,length)[0]
        lower = self.fc(lower)
        return lower

class high_module(nn.Module):
    def __init__(self):
        super(high_module, self).__init__()
        self.gpo=GPO(32,32)
        self.fc = nn.Linear(768,512)
    def forward(self, higher_feature):
        higher_feature = higher_feature.view(higher_feature.shape[0],higher_feature.shape[1],-1)
        # length=[384.]*higher_feature.shape[0]
        # length=torch.tensor(length).cuda() 
        higher = higher_feature.mean(dim=-1,keepdim=False)
        # higher= self.gpo(higher_feature,length)[0]
        higher = self.fc(higher)
        return higher












