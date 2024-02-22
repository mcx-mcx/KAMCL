# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .KAMCL_utils import *
import copy

from layers.gpo import GPO

class VisualModel(nn.Module):
    def __init__(self, opt={}):
        super(VisualModel, self).__init__()
        # img feature
        self.extract_feature = ExtractFeature(opt = opt) #TAG这个提取和之前的没区别
        self.gpo=GPO(32,32)
        self.low_module = low_module()
        self.high_module = high_module()
        
        self.mlp=nn.Sequential(nn.Linear(512, 512),nn.ReLU(),nn.Linear(512, 512),nn.ReLU())
        self.im = nn.Sequential(nn.Linear(512, 512),nn.Sigmoid())
        self.Eiters = 0

    def forward(self, img):
        lower_feature, higher_feature, solo_feature = self.extract_feature(img) 
        global_feature = solo_feature
        local_feature = self.low_module(lower_feature)
        higher_feature = self.high_module(higher_feature)
        tem=torch.cat([global_feature.unsqueeze(1),local_feature.unsqueeze(1),higher_feature.unsqueeze(1)],1)
        tem2 = self.mlp(tem)
        tem3 = torch.cat([tem2,tem],dim=1)
        length=[6.]*img.shape[0]
        length=torch.tensor(length).cuda() 
        visual_feature=self.gpo(tem3,length)[0]
        visual_concept = self.im(visual_feature)
        visual_feature = visual_feature + visual_concept
        return visual_feature,visual_concept

class TextModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(TextModel, self).__init__()
       
        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )
        self.tx = nn.Sequential(nn.Linear(512, 512),nn.Sigmoid())
        self.gpo = GPO(32,32)
        self.linear = nn.Linear(512,512)

    def forward(self, text,text_onehot, text_lens=None):
        text_feature = self.text_feature(text)  
        text_onehot= text_onehot.cuda()
        text_onehot = self.linear(text_onehot)
        tem = torch.cat([text_feature.unsqueeze(1),text_onehot.unsqueeze(1)],dim=1)     
        length=[2.]*text_feature.shape[0]
        length=torch.tensor(length).cuda() 
        text_feature = self.gpo(tem,length)[0]
        textconcept = self.tx(text_feature)
        text_feature =text_feature +textconcept
        return text_feature,textconcept


class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[], dim=512,K=8000, m=0.999, T=0.07):
        super(BaseModel, self).__init__()

        self.visualmodelQ=VisualModel(opt)
        self.textmodelQ=TextModel(opt,vocab_words)

        self.visualmodelK=VisualModel(opt)
        self.textmodelK=TextModel(opt,vocab_words)

        self.mlp=nn.Sequential(nn.Linear(512, 512), nn.ReLU(),nn.Linear(512, 512), nn.ReLU())
        
        self.im = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512)) 
        self.tx = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512)) #, nn.BatchNorm1d(512)

        self.K = K
        self.m = m
        self.T = T    
        self.Eiters = 0

        for param_vq, param_vk,param_tq,param_tk in zip(
            self.visualmodelQ.parameters(), self.visualmodelK.parameters(),
            self.textmodelQ.parameters(),self.textmodelK.parameters()
        ):
            param_vk.data.copy_(param_vq.data)  # initialize
            param_tk.data.copy_(param_tq.data)
            param_vk.requires_grad = False  # not update by gradient
            param_tk.requires_grad = False

        self.register_buffer("queue_im", torch.randn(dim, K))
        self.queue_im = nn.functional.normalize(self.queue_im, dim=0)
        self.register_buffer("queue_ptr_im", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_tx", torch.randn(dim, K))
        self.queue_tx = nn.functional.normalize(self.queue_tx, dim=0)
        self.register_buffer("queue_ptr_tx", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_vq, param_vk,param_tq,param_tk in zip(
            self.visualmodelQ.parameters(), self.visualmodelK.parameters(),
            self.textmodelQ.parameters(),self.textmodelK.parameters()
        ):
            param_vk.data = param_vk.data * self.m + param_vq.data * (1.0 - self.m)
            param_tk.data = param_tk.data * self.m + param_tq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_im(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_im)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_im[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_im[0] = ptr   

    @torch.no_grad()
    def _dequeue_and_enqueue_tx(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_tx)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_tx[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_tx[0] = ptr    

    def forward(self,epoch, img, text, text_onehot,text_lens=None):

        
        visual_feature,visual_feature_concept = self.visualmodelQ(img)      # queries: NxC
        text_feature,text_feature_concept = self.textmodelQ(text, text_onehot,text_lens)       
        
        if self.training:
            vq = l2norm(visual_feature, dim=-1)
            tq = l2norm(text_feature, dim=-1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                vk,_ = self.visualmodelK(img)  # keys: NxC
                tk,_ = self.textmodelK(text,text_onehot, text_lens) # keys: NxC

                vk = l2norm(vk, dim=-1)
                tk = l2norm(tk, dim=-1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos_v = torch.einsum("nc,nc->n", [vq, tk]).unsqueeze(-1)
            l_pos_t = torch.einsum("nc,nc->n", [tq, vk]).unsqueeze(-1)
            # negative logits: NxK
            l_neg_v = torch.einsum("nc,ck->nk", [vq, self.queue_tx.clone().detach()])
            l_neg_t = torch.einsum("nc,ck->nk", [tq, self.queue_im.clone().detach()])

            # logits: Nx(1+K)
            logits_im = torch.cat([l_pos_v, l_neg_v], dim=1)
            logits_tx = torch.cat([l_pos_t, l_neg_t], dim=1)

            # apply temperature
            logits_im /= self.T
            logits_tx /= self.T

            # # labels: positive key indicators
            labels_im = torch.zeros(logits_im.shape[0], dtype=torch.long).cuda()
            labels_tx = torch.zeros(logits_tx.shape[0], dtype=torch.long).cuda()
            # # dequeue and enqueue
 
            self._dequeue_and_enqueue_im(vk)
            self._dequeue_and_enqueue_tx(tk)
            
            sims = cosine_sim(visual_feature, text_feature)
        

            return sims,visual_feature_concept,text_feature_concept,logits_im,logits_tx,labels_im,labels_tx
            
        else:
            sims = cosine_sim(visual_feature, text_feature)
            return sims,visual_feature,text_feature

def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model



