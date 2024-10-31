import torch
from torch import nn
import cv2
import random
import numpy as np
from utils.steg_tools import embd_sim, uniward, hill
import models
import scipy.io as sci
from scipy.linalg import lstsq
from scipy.signal import convolve2d as con2d
from adv_emb import compute_fea_ig2

def UGS(cover, 
        model, 
        rho, 
        payload, 
        criterion = nn.CrossEntropyLoss(), 
        max_beta = 1.0, 
        beta_step = 0.1, 
        wetCost = 1e13):

    # init
    # if rho is 'SUNIWARD' or rho is None:
    #     rho = uniward(cover)
    # elif rho is 'hill':
    #     rho = hill(cover)
    # else:
    rhoP1 = np.reshape(rho[0], cover.size)
    rhoM1 = np.reshape(rho[1], cover.size)

    flat_rhoP1 = np.reshape(rhoP1, -1)
    sort_rhoP1 = np.sort(flat_rhoP1)

    flat_rhoM1 = np.reshape(rhoM1, -1)
    sort_rhoM1 = np.sort(flat_rhoM1)


    f1, f2, f3 = cal_filter(cover)
    # fs = np.stack(f1, f2, f3)
    # fs = np.expand_dims(fs, 1)
    # fs = fs.astype(np.float32)
    # fs = torch.from_numpy(fs)


    filters = torch.cat([torch.Tensor(f1).unsqueeze(0),torch.Tensor(f2).unsqueeze(0),torch.Tensor(f3).unsqueeze(0)],dim=0)
    # import ipdb
    # ipdb.set_trace()

    

    # cover_res = torch.conv2d(cover, f1)
    # cover_res = torch.nn.functional.conv2d(cover, f1)
    cover_res = torch.nn.functional.conv2d(torch.from_numpy(cover).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())

    img_len = cover.size
    msg_len = int(payload*img_len)
    # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
    cover_1d = cover.flatten()

    ini_stego = embd_sim(cover_1d, rhoP1.flatten(), rhoM1.flatten(), msg_len).reshape(cover.shape)
    ini_stego_res = torch.nn.functional.conv2d(torch.from_numpy(ini_stego).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())

    # compute gradient
    ref_cover = torch.from_numpy(np.reshape(cover_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
    ref_cover.requires_grad = True
    gradient = torch.autograd.grad(criterion(model(ref_cover)[0], torch.tensor([0],dtype=torch.int64).cuda()), ref_cover)[0]
    gradient = gradient.flatten().cpu().numpy()

    flat_grad = np.reshape(abs(gradient), -1)
    sort_grad = np.sort(flat_grad)

    positive_grad = gradient > 0
    negetive_grad = gradient < 0


    ini = torch.from_numpy(np.reshape(ini_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
    final_stego = ini_stego.copy()
    if torch.argmax(model(ini)[0],1) == 0:
        distance = torch.sum(torch.abs(ini_stego_res - cover_res), [1,2,3])
    else:
        distance = 1e13


    for grad_p in np.arange(0.1, 0.91, 0.1):
        for rho_p in np.arange(0.1, 0.91, 0.1):
            adv_rhoP1 = rhoP1.copy()
            adv_rhoM1 = rhoM1.copy()

            thr_grad = sort_grad[int(np.round(grad_p * 256 * 256))]
            thr_rhoP1 = sort_rhoP1[int(np.round(rho_p * 256 * 256))]
            thr_rhoM1 = sort_rhoM1[int(np.round(rho_p * 256 * 256))]

            high_grad = abs(gradient) > thr_grad
            small_prerhoP1 = rhoP1 < thr_rhoP1
            small_prerhoM1 = rhoM1 < thr_rhoM1

            adjust_rhoP1 = high_grad * small_prerhoP1 * positive_grad
            adjust_rhoM1 = high_grad * small_prerhoM1 * negetive_grad


            adv_rhoP1[adjust_rhoP1] = rhoP1[adjust_rhoP1]*2
            adv_rhoM1[adjust_rhoM1] = rhoM1[adjust_rhoM1]*2

            adv_rhoP1[adv_rhoP1>wetCost] = wetCost
            adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
            adv_rhoM1[adv_rhoM1>wetCost] = wetCost
            adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost

            adv_stego = embd_sim(cover_1d, adv_rhoP1.flatten(), adv_rhoM1.flatten(), msg_len).reshape(cover.shape)
            adv = torch.from_numpy(np.reshape(adv_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
            if torch.argmax(model(adv)[0],1) == 0:
                stego_res = torch.nn.functional.conv2d(torch.from_numpy(np.reshape(adv_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())
                # stego_res = torch.nn.functional.conv2d(cover,filters.unsqueeze(1).to(cover.device))
                tmp = torch.sum(torch.abs(stego_res - cover_res), [1,2,3])
                if tmp < distance:
                    distance = tmp
                    final_stego = adv_stego.copy()
                    
                # import ipdb
                # ipdb.set_trace()
    final = torch.from_numpy(np.reshape(final_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
    if torch.argmax(model(final)[0],1) == 0:
        return final_stego, 1, 0
    else:
        return final_stego, 0, 0



def NAA_UGS(cover, 
        model, 
        rho, 
        payload, 
        criterion = nn.CrossEntropyLoss(), 
        max_beta = 1.0, 
        beta_step = 0.1, 
        wetCost = 1e13):

    # init
    # if rho is 'SUNIWARD' or rho is None:
    #     rho = uniward(cover)
    # elif rho is 'hill':
    #     rho = hill(cover)
    # else:
    rhoP1 = np.reshape(rho[0], cover.size)
    rhoM1 = np.reshape(rho[1], cover.size)

    flat_rhoP1 = np.reshape(rhoP1, -1)
    sort_rhoP1 = np.sort(flat_rhoP1)

    flat_rhoM1 = np.reshape(rhoM1, -1)
    sort_rhoM1 = np.sort(flat_rhoM1)


    f1, f2, f3 = cal_filter(cover)
    # fs = np.stack(f1, f2, f3)
    # fs = np.expand_dims(fs, 1)
    # fs = fs.astype(np.float32)
    # fs = torch.from_numpy(fs)


    filters = torch.cat([torch.Tensor(f1).unsqueeze(0),torch.Tensor(f2).unsqueeze(0),torch.Tensor(f3).unsqueeze(0)],dim=0)
    # import ipdb
    # ipdb.set_trace()

    

    # cover_res = torch.conv2d(cover, f1)
    # cover_res = torch.nn.functional.conv2d(cover, f1)
    cover_res = torch.nn.functional.conv2d(torch.from_numpy(cover).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())

    img_len = cover.size
    msg_len = int(payload*img_len)
    # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
    cover_1d = cover.flatten()

    ini_stego = embd_sim(cover_1d, rhoP1.flatten(), rhoM1.flatten(), msg_len).reshape(cover.shape)
    ini_stego_res = torch.nn.functional.conv2d(torch.from_numpy(ini_stego).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())

    # compute gradient
    ref_cover = torch.from_numpy(np.reshape(cover_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
    ref_cover.requires_grad = True

    # ini_loss = criterion(model(ref_cover)[0], torch.tensor([0],dtype=torch.int64).cuda())
    
    ref_cover = ref_cover.squeeze(0)
    inputs = ref_cover.cpu().detach().numpy()
    ref = np.ones(inputs.shape)
    label_inputs = 1
    integrated_inputs = compute_fea_ig2(ref, inputs, label_inputs, model, 50, 0.5) # out_shape:feature shape
    gradient = integrated_inputs

    # gradient = torch.autograd.grad(criterion(model(ref_cover)[0], torch.tensor([0],dtype=torch.int64).cuda()), ref_cover)[0]
    gradient = gradient.flatten().cpu().numpy()

    flat_grad = np.reshape(abs(gradient), -1)
    sort_grad = np.sort(flat_grad)

    positive_grad = gradient > 0
    negetive_grad = gradient < 0


    ini = torch.from_numpy(np.reshape(ini_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
    final_stego = ini_stego.copy()
    if torch.argmax(model(ini)[0],1) == 0:
        distance = torch.sum(torch.abs(ini_stego_res - cover_res), [1,2,3])
    else:
        distance = 1e13


    for grad_p in np.arange(0.1, 0.91, 0.1):
        for rho_p in np.arange(0.1, 0.91, 0.1):
            adv_rhoP1 = rhoP1.copy()
            adv_rhoM1 = rhoM1.copy()

            thr_grad = sort_grad[int(np.round(grad_p * 256 * 256))]
            thr_rhoP1 = sort_rhoP1[int(np.round(rho_p * 256 * 256))]
            thr_rhoM1 = sort_rhoM1[int(np.round(rho_p * 256 * 256))]

            high_grad = abs(gradient) > thr_grad
            small_prerhoP1 = rhoP1 < thr_rhoP1
            small_prerhoM1 = rhoM1 < thr_rhoM1

            adjust_rhoP1 = high_grad * small_prerhoP1 * positive_grad
            adjust_rhoM1 = high_grad * small_prerhoM1 * negetive_grad


            adv_rhoP1[adjust_rhoP1] = rhoP1[adjust_rhoP1]*2
            # adv_rhoP1[adjust_rhoM1] = rhoP1[adjust_rhoM1]/2
            adv_rhoM1[adjust_rhoM1] = rhoM1[adjust_rhoM1]*2
            # adv_rhoM1[adjust_rhoP1] = rhoM1[adjust_rhoP1]/2

            adv_rhoP1[adv_rhoP1>wetCost] = wetCost
            adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
            adv_rhoM1[adv_rhoM1>wetCost] = wetCost
            adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost

            adv_stego = embd_sim(cover_1d, adv_rhoP1.flatten(), adv_rhoM1.flatten(), msg_len).reshape(cover.shape)
            adv = torch.from_numpy(np.reshape(adv_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
            if torch.argmax(model(adv)[0],1) == 0:
                stego_res = torch.nn.functional.conv2d(torch.from_numpy(np.reshape(adv_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0), filters.unsqueeze(1).cuda())
                # stego_res = torch.nn.functional.conv2d(cover,filters.unsqueeze(1).to(cover.device))
                tmp = torch.sum(torch.abs(stego_res - cover_res), [1,2,3])
                if tmp < distance:
                    distance = tmp
                    final_stego = adv_stego.copy()
                    
                # import ipdb
                # ipdb.set_trace()
    final = torch.from_numpy(np.reshape(final_stego, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
    if torch.argmax(model(final)[0],1) == 0:
        return final_stego, 1, 0
    else:
        return final_stego, 0, 0

    # for beta in np.arange(0, max_beta + 0.01, beta_step):
    #     # recover the stego for later usage
    #     stego_1d = cover_1d.copy()

    #     # divide
    #     conv_ind = rand_ind[0 : int((1-beta)*img_len)]
    #     adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
    #     conv_cover = cover_1d[conv_ind]
    #     adv_cover = cover_1d[adv_ind]
    #     conv_rhoP1 = rhoP1[conv_ind].copy()
    #     conv_rhoM1 = rhoM1[conv_ind].copy()
    #     adv_rhoP1 = rhoP1[adv_ind].copy()
    #     adv_rhoM1 = rhoM1[adv_ind].copy()
    #     conv_rhoP1[conv_cover==255] = wetCost
    #     conv_rhoM1[conv_cover==0] = wetCost
    #     adv_rhoP1[adv_cover==255] = wetCost
    #     adv_rhoM1[adv_cover==0] = wetCost

    #     # embed conventional
    #     if beta != 1.0:
    #         conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
    #         stego_1d[conv_ind] = conv_stego

    #     # calculate the gradients
    #     model.zero_grad()
    #     if isinstance(criterion, models.SiaStegNetLoss):
    #         ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
    #                      torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
    #         ref_stego[0].requires_grad = True
    #         ref_stego[1].requires_grad = True
    #         cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
    #         loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
    #         loss.backward()
    #         gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
    #     else:
    #         ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
    #         ref_stego.requires_grad = True
    #         gradient = torch.autograd.grad(criterion(model(ref_stego)[0], torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
    #     gradient = gradient.flatten()[adv_ind].cpu().numpy()

        # import ipdb
        # ipdb.set_trace()
        
        # adjust the distortion
    #     adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
    #     adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
    #     adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
    #     adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
    #     adv_rhoP1[adv_rhoP1>wetCost] = wetCost
    #     adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
    #     adv_rhoM1[adv_rhoM1>wetCost] = wetCost
    #     adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost


    #     if beta != 0:
    #         adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
    #         stego_1d[adv_ind] = adv_stego.copy()

    #     if isinstance(criterion, models.SiaStegNetLoss):
    #         adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
    #                           torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
    #         if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
    #             attack_num = round(beta*10)+1
    #             return np.reshape(stego_1d, cover.shape), 1, attack_num
    #     else:
    #         adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
    #         if torch.argmax(model(adv_stego)[0],1) == 0:
    #             attack_num = round(beta*10)+1
    #             return np.reshape(stego_1d, cover.shape), 1, attack_num

    # attack_num = round(beta*10)+1
    # conv_rhoP1 = rhoP1.copy()
    # conv_rhoM1 = rhoM1.copy()
    # conv_rhoP1[conv_cover==255] = wetCost
    # conv_rhoM1[conv_cover==0] = wetCost
    # conv_cover = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)
    # return conv_cover, 0, attack_num


def cal_filter(img):
    f_n = 7
    block_shape = (1, f_n)

    col_1 = im2col(img, block_shape)
    col_2 = im2col(img.T, block_shape)
    col = np.concatenate([col_1, col_2], axis=1)
    neighbor = np.concatenate([col[:f_n//2], col[f_n//2+1:]], axis=0)
    target = col[f_n//2]

    sol = lstsq(neighbor.T, target.T)[0]

    base_f = -np.ones(f_n)
    base_f[:f_n//2] = sol[:f_n//2]
    base_f[f_n//2+1:] = sol[f_n//2:]
    # import ipdb
    # ipdb.set_trace()

    f_array = np.zeros((f_n, f_n, 2))
    f_array[1,:,0]=base_f
    f1 = f_array[:,:,0]
    f_array[:,1,1]=base_f
    f2 = f_array[:,:,1]
    base = base_f.reshape(1,-1)
    f3 = con2d(base, base.T)

    return f1, f2, f3




def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result

