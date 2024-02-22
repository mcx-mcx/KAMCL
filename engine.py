#encoding:utf-8
# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
from thop import profile
import logging
from torch.nn.utils.clip_grad import clip_grad_norm

def train(train_loader, model, optimizer, epoch, opt={}):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        torch.cuda.empty_cache()
        images,  captions, lengths, ids,imagetag ,text_onehot= train_data

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)


        input_text = Variable(captions)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()

            imagetag = imagetag.cuda()
            input_text = input_text.cuda()
            text_onehot = text_onehot.cuda()
        scores,visual_feature,text_feature,logits_im,logits_tx,labels_im,labels_tx = model(epoch,input_visual, input_text ,text_onehot)
        torch.cuda.synchronize()
        loss,cost,infoNCE_im_tx,BCEloss= utils.calcul_loss(epoch,imagetag,scores, input_visual.size(0), margin,visual_feature,text_feature,ids, logits_im,logits_tx,labels_im,labels_tx,max_violation=max_violation, )

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        train_logger.update('L', loss.cpu().data.numpy())
        train_logger.update('cost', cost.cpu().data.numpy())
        train_logger.update('infoNCE_im_tx', infoNCE_im_tx.cpu().data.numpy())
        train_logger.update('BCEloss', BCEloss.cpu().data.numpy())

        optimizer.zero_grad()  
        loss.backward()
        torch.cuda.synchronize() 
        optimizer.step()        
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )


def validate(val_loader, model):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 512),dtype='float32')


    input_text = np.zeros((len(val_loader.dataset), 512),dtype='float32')

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):

            images, captions, lengths, ids,imagetag,text_onehot = val_data
            images = images.cuda()
            captions = captions.cuda()
            text_onehot = text_onehot.cuda()
            sims,visual_feature,text_feature= model(1,images,captions,text_onehot,lengths)
            for (id, img, cap) in zip(ids, (visual_feature.cpu().numpy().copy()), (text_feature.cpu().numpy().copy())):
                input_visual[id] = img
                input_text[id] = cap



    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual,  input_text, lengths=None)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    #currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0
    rsum = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0
    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, rsum
    )
  


    return currscore, all_score


def validate_test(val_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    
    input_visual = np.zeros((len(val_loader.dataset), 512),dtype='float32')
    input_text = np.zeros((len(val_loader.dataset), 512),dtype='float32')
    # input_text_lengeth = [0]*len(val_loader.dataset)
    # input_text_onehot = np.zeros((len(val_loader.dataset), 512), dtype=np.float32)
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):

            images, captions, lengths, ids,imagetag,text_onehot = val_data
            images = images.cuda()
            captions = captions.cuda()
            text_onehot = text_onehot.cuda()
            sims,visual_feature,text_feature= model(1,images,captions,text_onehot,lengths)
            for (id, img, cap) in zip(ids, (visual_feature.cpu().numpy().copy()), (text_feature.cpu().numpy().copy())):
                input_visual[id] = img
                input_text[id] = cap



    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual,  input_text, lengths=None)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d


