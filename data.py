# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image
import json

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']
        self.data_split = data_split
        # Captions
        self.captions = []
        self.maxlength = 0
        self.tag_vocab_list= json.load(open(opt['dataset']['tag_vocab'], 'r'))
        self.tag2idx = dict(zip(self.tag_vocab_list, range(512)))

        if data_split == 'train':
            self.imageid2tags = {}
            self.tag_vocab_list= json.load(open(opt['dataset']['tag_vocab'], 'r'))
            self.tag2idx = dict(zip(self.tag_vocab_list, range(512)))

            for line in open(opt['dataset']['tag_path']).readlines():
                # print(line)
                if len(line.strip().split("\t", 1)) < 2:  # no tag available for a specific video
                    imageid = line.strip().split("\t", 1)[0]
                    self.imageid2tags[imageid] = []
                else:
                    imageid, or_tags = line.strip().split("\t", 1)
                    tags = [x.split(':')[0] for x in or_tags.strip().split()]
                    
                    # weighed concept scores
                    scores = [float(x.split(':')[1]) for x in or_tags.strip().split()]
                    scores = np.array(scores) / max(scores)

                    self.imageid2tags[imageid] = list(zip(tags, scores))
                    
        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []



            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
         


                    self.images.append(line.strip())
            
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []

            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                   


                    self.images.append(line.strip())
 
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab
        text_emb =512
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        text_onehot = torch.zeros(text_emb)
        text_list = [self.tag2idx[word] for word in tokens_UNK if word in self.tag2idx]
        for idx, tag_idx in enumerate(text_list):
           text_onehot[tag_idx] = 1
        text_onehot = torch.Tensor(np.array(text_onehot))

        #XXX target
        if self.data_split == 'train':
            imageid_tag_str = self.imageid2tags[str(img_id)]     # string representation
            tag_in_vocab = [tag_score for tag_score in imageid_tag_str if tag_score[0] in self.tag2idx]
            tag_list = [self.tag2idx[tag_score[0]] for tag_score in tag_in_vocab ]  # index representation
            score_list = [tag_score[1] for tag_score in tag_in_vocab]
            tag_one_hot = torch.zeros(text_emb)  # build zero vector of tag vocabulary that is used to represent tags by one-hot
            for idx, tag_idx in enumerate(tag_list):
                tag_one_hot[tag_idx] = score_list[idx]  # one-hot
            imagetag = torch.Tensor(np.array(tag_one_hot))
        else :
            tag_one_hot = torch.zeros(text_emb)
            imagetag = torch.Tensor(np.array(tag_one_hot))
        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])



        return image,  caption, tokens_UNK, index, img_id, imagetag,text_onehot  

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, tokens, ids, img_ids,imagetag,text_onehot = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    imagetag = torch.stack(imagetag,0)

    text_onehot = torch.stack(text_onehot, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids,imagetag,text_onehot


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, droplast=True,num_workers=0, opt={},):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=droplast)
    return data_loader

def get_loaders(vocab, opt):
    train_loader = get_precomp_loader( 'train', vocab,
                                      opt['dataset']['batch_size'], True, True,opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt['dataset']['batch_size_val'], False, False,opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader( 'test', vocab,
                                      opt['dataset']['batch_size_val'], False,False, opt['dataset']['workers'], opt=opt)
    return test_loader
