import torch
from torch import nn
import cv2
import random
import numpy as np
from utils.steg_tools import embd_sim
import models

def SPS_ENH(cover, 
        model, 
        rho, 
        payload, 
        criterion = nn.CrossEntropyLoss(), 
        max_beta = 1.0, 
        beta_step = 0.1, 
        wetCost = 1e13):

    max_iter = 500
    restart_iter = 10
    delta = 1
    alpha = 1

    # init
    # if rho is 'SUNIWARD' or rho is None:
    #     rho = uniward(cover)
    # elif rho is 'hill':
    #     rho = hill(cover)
    # else:
    rhoP1 = np.reshape(rho[0], cover.size)
    rhoM1 = np.reshape(rho[1], cover.size)


    img_len = cover.size
    msg_len = payload*img_len

    conv_stego = embd_sim(cover.flatten(), rhoP1, rhoM1, int(msg_len))
    conv_stego = np.reshape(conv_stego, cover.shape)
    pseudo_cover = conv_stego.copy()
    pseudo_cover_ph = pseudo_cover.copy()

    ori_pred_label = torch.argmax(model(torch.from_numpy(conv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0], 1)
    if ori_pred_label == 0:
        return pseudo_cover, 1, 0
    mod_pos = np.zeros(pseudo_cover.shape)
    mod_num = 0
    for iter_idx in range(max_iter):
        mod_num += delta
        pseudo_cover_ph = pseudo_cover.copy()
        pseudo_cover_ph = torch.from_numpy(pseudo_cover_ph).float().cuda().unsqueeze(0).unsqueeze(0)
        pseudo_cover_ph.requires_grad = True
        dy_dp_v = torch.autograd.grad(criterion(model(pseudo_cover_ph)[0], torch.tensor([1],dtype=torch.int64).cuda()), pseudo_cover_ph)[0]
        mod_pos_vec = mod_pos.reshape(img_len)
        # import ipdb
        # ipdb.set_trace()

        grad_vec = dy_dp_v.cpu().detach().squeeze(0).squeeze(0).numpy()*(1-mod_pos)
        mod_pos_vec[np.argsort(np.abs(grad_vec.reshape(256*256)))[-mod_num:]] = 1
        mod_pos = mod_pos_vec.reshape((256, 256))
        delta_map = alpha*np.sign(grad_vec*mod_pos)
        pseudo_cover += delta_map.reshape((256,256))
        pseudo_cover = np.clip(np.round(pseudo_cover), 0, 255)
        enh_cover = np.clip(cover + pseudo_cover - conv_stego, 0, 255).flatten()
        rhoP1, rhoM1 = rhoP1.copy(), rhoM1.copy()
        rhoP1[enh_cover==255], rhoM1[enh_cover==0] = wetCost, wetCost
        adv_stego = embd_sim(enh_cover.flatten(), rhoP1, rhoM1, int(msg_len))
        adv_stego = np.reshape(adv_stego, cover.shape)
        prob = model(torch.from_numpy(adv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0]
        # import ipdb
        # ipdb.set_trace()

        if prob[0][0].item() > 0.5:
            return adv_stego, 1, 0
        elif prob[0][0].item() > 0.01:
            for repeat_idx in range(restart_iter):
                adv_stego = embd_sim(enh_cover.flatten(), rhoP1, rhoM1, int(msg_len))
                adv_stego = np.reshape(adv_stego, cover.shape)
                prob = model(torch.from_numpy(adv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0]
                if prob[0][0].item() > 0.5:
                    return adv_stego, 1, 0
        if iter_idx == max_iter - 1 and prob[0][0].item() < 0.5:
            return conv_stego, 0, 0


def NAA_SPS_ENH(cover, 
        model, 
        rho, 
        payload, 
        criterion = nn.CrossEntropyLoss(), 
        max_beta = 1.0, 
        beta_step = 0.1, 
        wetCost = 1e13):

    max_iter = 500
    restart_iter = 10
    delta = 1
    alpha = 1

    # init
    # if rho is 'SUNIWARD' or rho is None:
    #     rho = uniward(cover)
    # elif rho is 'hill':
    #     rho = hill(cover)
    # else:
    rhoP1 = np.reshape(rho[0], cover.size)
    rhoM1 = np.reshape(rho[1], cover.size)


    img_len = cover.size
    msg_len = payload*img_len

    conv_stego = embd_sim(cover.flatten(), rhoP1, rhoM1, int(msg_len))
    conv_stego = np.reshape(conv_stego, cover.shape)
    pseudo_cover = conv_stego.copy()
    pseudo_cover_ph = pseudo_cover.copy()

    ori_pred_label = torch.argmax(model(torch.from_numpy(conv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0], 1)
    if ori_pred_label == 0:
        return pseudo_cover, 1, 0
    mod_pos = np.zeros(pseudo_cover.shape)
    mod_num = 0
    for iter_idx in range(max_iter):
        mod_num += delta
        pseudo_cover_ph = pseudo_cover.copy()
        pseudo_cover_ph = torch.from_numpy(pseudo_cover_ph).float().cuda().unsqueeze(0).unsqueeze(0)
        # pseudo_cover_ph.requires_grad = True

        # ini_loss = criterion(model(pseudo_cover_ph)[0], torch.tensor([1],dtype=torch.int64).cuda())

        pseudo_cover_ph = pseudo_cover_ph.squeeze(0)
        inputs = pseudo_cover_ph.cpu().detach().numpy()
        label_inputs = 1
        integrated_inputs = compute_fea_ig(inputs, label_inputs, model, 30, 0.3) # out_shape:feature shape
        dy_dp_v = integrated_inputs

        # dy_dp_v = torch.autograd.grad(criterion(model(pseudo_cover_ph)[0], torch.tensor([1],dtype=torch.int64).cuda()), pseudo_cover_ph)[0]
        mod_pos_vec = mod_pos.reshape(img_len)
        # import ipdb
        # ipdb.set_trace()

        grad_vec = dy_dp_v.cpu().detach().squeeze(0).squeeze(0).numpy()*(1-mod_pos)
        mod_pos_vec[np.argsort(np.abs(grad_vec.reshape(256*256)))[-mod_num:]] = 1
        mod_pos = mod_pos_vec.reshape((256, 256))
        delta_map = alpha*np.sign(grad_vec*mod_pos)
        pseudo_cover += delta_map.reshape((256,256))
        pseudo_cover = np.clip(np.round(pseudo_cover), 0, 255)
        enh_cover = np.clip(cover + pseudo_cover - conv_stego, 0, 255).flatten()
        rhoP1, rhoM1 = rhoP1.copy(), rhoM1.copy()
        rhoP1[enh_cover==255], rhoM1[enh_cover==0] = wetCost, wetCost
        adv_stego = embd_sim(enh_cover.flatten(), rhoP1, rhoM1, int(msg_len))
        adv_stego = np.reshape(adv_stego, cover.shape)
        prob = model(torch.from_numpy(adv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0]
        # import ipdb
        # ipdb.set_trace()

        if prob[0][0].item() > 0.5:
            return adv_stego, 1, 0
        elif prob[0][0].item() > 0.01:
            for repeat_idx in range(restart_iter):
                adv_stego = embd_sim(enh_cover.flatten(), rhoP1, rhoM1, int(msg_len))
                adv_stego = np.reshape(adv_stego, cover.shape)
                prob = model(torch.from_numpy(adv_stego).float().cuda().unsqueeze(0).unsqueeze(0))[0]
                if prob[0][0].item() > 0.5:
                    return adv_stego, 1, 0
        if iter_idx == max_iter - 1 and prob[0][0].item() < 0.5:
            return conv_stego, 0, 0

def compute_fea_ig(fea_loss, inputs,label_inputs,model, steps, thr):
    baseline = np.zeros(inputs.shape)
    weight = torch.zeros((1, 128, 64, 64)).cuda()

    scaled_inputs = []
    feas = []
    for idx in range(steps):

        
        co = inputs + np.random.normal(size = inputs.shape, loc=0.0, scale=0.25)
        co = torch.from_numpy(co).cuda().unsqueeze(0)
        co.requires_grad_(True)
        scaled_inputs.append(co)
        

        output, fea = model(co)
        feas.append(fea)
        fea_gradient = torch.autograd.grad(output[:, label_inputs], fea, grad_outputs=torch.ones_like(output[:, label_inputs]), retain_graph=True)[0]
        x_gradient = torch.autograd.grad(output[:, label_inputs], co, grad_outputs=torch.ones_like(output[:, label_inputs]), retain_graph=True)[0]
        inputs = inputs + 0.0025 * np.sign(x_gradient.detach().cpu()).squeeze(0).numpy()
        weight = weight + fea_gradient
    

    IA = weight / steps


    delta_Y = feas[-1] - feas[0]
    integrated_grad = delta_Y * IA

    IG = integrated_grad.unsqueeze(0)
    tmp_sum = torch.sum(IG, (-1, -2))
    sort_sum, _ = torch.sort(abs(tmp_sum))
    T = sort_sum[0, -int(sort_sum.shape[1]*thr)]
    position = torch.where(abs(tmp_sum)>T)[1]

    loss = torch.sum(tmp_sum[0, position])

    model.zero_grad()
    loss.backward()
    avg_grads =  scaled_inputs[0].grad.data +  scaled_inputs[-1].grad.data
    # del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
    return avg_grads
