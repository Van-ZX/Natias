import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
import cv2
# import logging
# logger = logging.getLogger(__name__)

from adv_emb import *
from utils import CoverRhoDataset
import models
from models.CovNet import CovNet
from models.SRNet import SRNet
from LWENet import lwenet

def parse_args():
    parser = argparse.ArgumentParser()
    """
    To align with the settings of SiaStegNet,
    the train/valid/test datasets are put in different directories
    """
    parser.add_argument('--train_cover_dir', type=str, required=True,)
    parser.add_argument('--val_cover_dir', type=str, required=True,)
    parser.add_argument('--test_cover_dir', type=str, required=True,)
    parser.add_argument('--train_rho_dir', type=str, required=True,)
    parser.add_argument('--val_rho_dir', type=str, required=True,)
    parser.add_argument('--test_rho_dir', type=str, required=True,)
    parser.add_argument('--adv_stego_dir', type=str, required=True,)
    parser.add_argument('--batch_size', type=int, default=50,) # only used for dataloader
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model', type=str, default='SiaStegNet')
    parser.add_argument('--ckpt_dir', type=str, required=True)

    parser.add_argument('--payload', type=float, default=0.4)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--beta_step', type=float, default=0.1)
    parser.add_argument('--wet_cost', type=float, default=1e13)

    args = parser.parse_args()
    return args

def build_model(args):
    if args.model == 'SiaStegNet':
        net = nn.DataParallel(models.KeNet())
        criterion = models.SiaStegNetLoss(margin=1.0,alpha=0.1)
        ckpt_path = args.ckpt_dir + '/model_best.pth.tar'
        ckpt = torch.load(ckpt_path)
        best_acc = ckpt['best_prec1']
        net.load_state_dict(ckpt['state_dict'])
        print('Load ' + args.model + ' from' + args.ckpt_dir)
        print('Acc on test dataset is: {:.4f}'.format(best_acc))
        return net.cuda(), criterion.cuda()
    elif args.model == 'SID':
        net = nn.DataParallel(models.SID())
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'XuNet':
        net = nn.DataParallel(models.XuNet())
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'YeNet':
        net = nn.DataParallel(models.YeNet())
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'CovNet':
        # net = nn.DataParallel(CovNet())
        net = CovNet()
        criterion = nn.CrossEntropyLoss()
        ckpt_path = args.ckpt_dir + '/model_params.pt'
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt['original_state'])
        print('Load ' + args.model + ' from' + args.ckpt_dir)
        return net.cuda(), criterion.cuda()
    elif args.model == 'SRNet':
        # net = nn.DataParallel(CovNet())
        net = SRNet()
        criterion = nn.CrossEntropyLoss()
        ckpt_path = args.ckpt_dir + '/model_params.pt'
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt['original_state'])
        print('Load ' + args.model + ' from' + args.ckpt_dir)
        return net.cuda(), criterion.cuda()
    elif args.model == 'LWENet':
        # net = nn.DataParallel(CovNet())
        net = lwenet()
        criterion = nn.CrossEntropyLoss()
        ckpt_path = args.ckpt_dir + '/model_params.pt'
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt['original_state'])
        print('Load ' + args.model + ' from' + args.ckpt_dir)
        return net.cuda(), criterion.cuda()
    else:
        raise NotImplementedError
    ckpt_path = args.ckpt_dir
    # ckpt_path = args.ckpt_dir + '/model_best.pth.tar'
    ckpt = torch.load(ckpt_path)
    # best_acc = ckpt['best_prec1']
    net.load_state_dict(ckpt['original_state'])
    print('Load ' + args.model + ' from' + args.ckpt_dir)
    # print('Acc on test dataset is: {:.4f}'.format(best_acc))
    return net.cuda(), criterion.cuda()

def build_dataloader(img_dir, rho_dir, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = CoverRhoDataset(
            img_dir = img_dir, 
            rho_dir = rho_dir, 
            # indices = range(11209, 14000),
            transform = transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers)
    return dataloader

def preprocess_data(images):
    # images of shape: NxCxHxW
    if images.dim() == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        # labels = labels.squeeze(0)
    h, w = images.shape[-2:]
    ch, cw, h0, w0 = h, w, 0, 0

    if args.model == 'SiaStegNet':
        cw = cw & ~1
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
        
    elif args.model == 'SID':
        inputs = [images[..., h0:h0 + ch, w0:w0 + cw]]

    # if args.cuda:
    inputs = [x.cuda() for x in inputs]
    # labels = labels.cuda()
    return inputs

def gen_adv_stego(dataloader, model, criterion, args, partion):
    if partion == 0:
        path = args.adv_stego_dir + 'train_stego_14000/'
    elif partion == 1:
        path = args.adv_stego_dir + 'valid_stego_1000/'
    elif partion == 2:
        path = args.adv_stego_dir + 'test_stego_5000/'
    else:
        print("partion error!\n")
    os.makedirs(path, exist_ok=True)

    model.eval()
    num_succ = 0
    sum_attack_num = 0
    sum_time = 0.0

    for idx, data in enumerate(dataloader):
        covers, rhos, names, labels = data[0].numpy()*255., data[1].numpy(), data[2], data[3].cuda()
        # import ipdb
        # ipdb.set_trace()

        for i in range(covers.shape[0]):
            # print("hello")
            start = time.time()
            adv_stego, succ, attack_num = Natias(
                        cover = covers[i,0,...], 
                        payload = args.payload, 
                        model = model, 
                        rho = rhos[i, :, ...], 
                        criterion = criterion)
            end = time.time()
            run_time = end - start
            num_succ += succ
            sum_attack_num += attack_num
            sum_time += run_time
            cv2.imwrite(path + names[i], adv_stego)
        if idx < dataloader.__len__()-1:
            print('The success number is: {}/{}'.format(num_succ, (idx+1)*dataloader.batch_size))
    print('The success number is: {}/{}'.format(num_succ, dataloader.dataset.len))
    print('attack queries: ' , sum_attack_num)
    print('attack time: ', sum_time)

args = parse_args()
os.makedirs(args.adv_stego_dir, exist_ok=True)
model, criterion = build_model(args)
train_loader = build_dataloader(
                args.train_cover_dir,
                args.train_rho_dir,
                args.batch_size,
                args.num_workers)
val_loader = build_dataloader(
                args.val_cover_dir,
                args.val_rho_dir,
                args.batch_size,
                args.num_workers)
test_loader = build_dataloader(
                args.test_cover_dir,
                args.test_rho_dir,
                args.batch_size,
                args.num_workers)

gen_adv_stego(test_loader, model, criterion, args, 2)
gen_adv_stego(val_loader, model, criterion, args, 1)
gen_adv_stego(train_loader, model, criterion, args, 0)

