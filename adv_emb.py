import torch
from torch import nn
import cv2
import random
import numpy as np
from utils.steg_tools import embd_sim, uniward, hill
import models
import scipy.io as sci

def adv_embed(cover, 
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


    img_len = cover.size
    msg_len = payload*img_len
    # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
    rand_ind = np.array(random.sample(range(img_len), k = img_len))
    cover_1d = cover.flatten()

    for beta in np.arange(0, max_beta + 0.01, beta_step):
        # recover the stego for later usage
        stego_1d = cover_1d.copy()

        # divide
        conv_ind = rand_ind[0 : int((1-beta)*img_len)]
        adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
        conv_cover = cover_1d[conv_ind]
        adv_cover = cover_1d[adv_ind]
        conv_rhoP1 = rhoP1[conv_ind].copy()
        conv_rhoM1 = rhoM1[conv_ind].copy()
        adv_rhoP1 = rhoP1[adv_ind].copy()
        adv_rhoM1 = rhoM1[adv_ind].copy()
        conv_rhoP1[conv_cover==255] = wetCost
        conv_rhoM1[conv_cover==0] = wetCost
        adv_rhoP1[adv_cover==255] = wetCost
        adv_rhoM1[adv_cover==0] = wetCost

        # embed conventional
        if beta != 1.0:
            conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
            stego_1d[conv_ind] = conv_stego

        # calculate the gradients
        model.zero_grad()
        if isinstance(criterion, models.SiaStegNetLoss):
            ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
                         torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
            ref_stego[0].requires_grad = True
            ref_stego[1].requires_grad = True
            cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
            loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
            loss.backward()
            gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
        else:
            ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
            ref_stego.requires_grad = True
            # import ipdb
            # ipdb.set_trace()
            gradient = torch.autograd.grad(criterion(model(ref_stego)[0], torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
        gradient = gradient.flatten()[adv_ind].cpu().numpy()

        # import ipdb
        # ipdb.set_trace()
        
        # adjust the distortion
        adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
        adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
        adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
        adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
        adv_rhoP1[adv_rhoP1>wetCost] = wetCost
        adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
        adv_rhoM1[adv_rhoM1>wetCost] = wetCost
        adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost


        if beta != 0:
            adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
            stego_1d[adv_ind] = adv_stego.copy()

        if isinstance(criterion, models.SiaStegNetLoss):
            adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
                              torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
            if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
                attack_num = round(beta*10)+1
                return np.reshape(stego_1d, cover.shape), 1, attack_num
        else:
            adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
            if torch.argmax(model(adv_stego)[0],1) == 0:
                attack_num = round(beta*10)+1
                return np.reshape(stego_1d, cover.shape), 1, attack_num

    attack_num = round(beta*10)+1
    conv_rhoP1 = rhoP1.copy()
    conv_rhoM1 = rhoM1.copy()
    conv_rhoP1[conv_cover==255] = wetCost
    conv_rhoM1[conv_cover==0] = wetCost
    conv_cover = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)
    return conv_cover, 0, attack_num


def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad


def Natias(cover, 
        model, 
        rho, 
        payload, 
        criterion = nn.CrossEntropyLoss(), 
        max_beta = 1.0, 
        beta_step = 0.1, 
        wetCost = 1e13):

    ens = 30

    
    
    rhoP1 = np.reshape(rho[0], cover.size)
    rhoM1 = np.reshape(rho[1], cover.size)
    # sorted_indices = np.argsort(rhoP1)

    img_len = cover.size
    msg_len = payload*img_len
    # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()

    rand_ind = np.array(random.sample(range(img_len), k = img_len))
    # rand_ind = sorted_indices[::-1]
    cover_1d = cover.flatten()

    for beta in np.arange(0, max_beta + 0.01, beta_step):
    # for beta in np.arange(0, 0.5 + 0.01, 0.05):
        # recover the stego for later usage
        stego_1d = cover_1d.copy()
        conv_ind = rand_ind[0 : int((1-beta)*img_len)]
        adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
        conv_cover = cover_1d[conv_ind]
        adv_cover = cover_1d[adv_ind]
        conv_rhoP1 = rhoP1[conv_ind].copy()
        conv_rhoM1 = rhoM1[conv_ind].copy()
        adv_rhoP1 = rhoP1[adv_ind].copy()
        adv_rhoM1 = rhoM1[adv_ind].copy()
        conv_rhoP1[conv_cover==255] = wetCost
        conv_rhoM1[conv_cover==0] = wetCost
        adv_rhoP1[adv_cover==255] = wetCost
        adv_rhoM1[adv_cover==0] = wetCost

        # embed conventional
        if beta != 1.0:
            conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
            stego_1d[conv_ind] = conv_stego
        
        ref = stego_1d.copy()
        ref[conv_ind] = 0
        ref[adv_ind] = 1
        ref = np.reshape(ref, cover.shape)

        # adv_stego = torch.from_numpy(np.reshape(cover, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
        # ref_stego = adv_stego.squeeze(0)
        # inputs = ref_stego.cpu().detach().numpy()
        # label_inputs = 1
        # # integrated_inputs = compute_fea_ig4(ref, inputs, label_inputs, model, 50, 0.5) # out_shape:feature shape
        # integrated_inputs = compute_ig(inputs, label_inputs, model, 50)
        # gradient = integrated_inputs
        # grad = gradient.cpu().detach().numpy().squeeze(0)
        # import ipdb
        # ipdb.set_trace()
        # sci.savemat("stego_NAA_Cov.mat",{"grad":grad})

        # calculate the gradients
        model.zero_grad()
        if isinstance(criterion, models.SiaStegNetLoss):
            ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
                         torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
            ref_stego[0].requires_grad = True
            ref_stego[1].requires_grad = True
            cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
            loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
            loss.backward()
            gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
        else:
            ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
            ref_stego.requires_grad = True
            #calculate the attribution
            # ini_loss = criterion(model(ref_stego)[0], torch.tensor([0],dtype=torch.int64).cuda())

            # stego_fea = model(ref_stego)[1]

            # ref_cover = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
            # ref_cover.requires_grad = True
            # cover_fea = model(ref_cover)[1]

            # up = stego_fea - cover_fea
            # down = cover_fea

            # fea_loss = torch.norm(up, p=2).item() / torch.norm(down, p=2).item()

            ref_stego = ref_stego.squeeze(0)
            inputs = ref_stego.cpu().detach().numpy()
            label_inputs = 1
            
            # # min loss

            integrated_inputs = compute_fea_ig4(ref, inputs, label_inputs, model, 50, 0.5) # out_shape:feature shape
            
            gradient = integrated_inputs

            # import ipdb
            # ipdb.set_trace()
            
            # attacker = ILPD(source_model = model)
            # gradient = attacker(ori_img = ref_stego, label = torch.tensor([0],dtype=torch.int64).cuda())

            # gradient = torch.autograd.grad(criterion(model(ref_stego), torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
            gradient = gradient.detach().cpu().numpy().flatten()[adv_ind]
            
            
            # gradient = torch.autograd.grad(attribution, torch.tensor([0], dtype=torch.int64).cuda(), ref_stego)[0]

        # adjust the distortion
        adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
        adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
        adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
        adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
        adv_rhoP1[adv_rhoP1>wetCost] = wetCost
        adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
        adv_rhoM1[adv_rhoM1>wetCost] = wetCost
        adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost


        if beta != 0:
            adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
            stego_1d[adv_ind] = adv_stego.copy()

        if isinstance(criterion, models.SiaStegNetLoss):
            adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
                              torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
            if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
                attack_num = round(beta*10)+1
                return np.reshape(stego_1d, cover.shape), 1, attack_num
        else:
            adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
            if torch.argmax(model(adv_stego)[0],1) == 0:
                attack_num = round(beta*10)+1
                return np.reshape(stego_1d, cover.shape), 1, attack_num

    attack_num = round(beta*10)+1
    conv_rhoP1 = rhoP1.copy()
    conv_rhoM1 = rhoM1.copy()
    conv_rhoP1[cover_1d==255] = wetCost
    conv_rhoM1[cover_1d==0] = wetCost
    conv_stego = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)

    return conv_stego, 0, attack_num



def compute_fea_ig4(ref, inputs,label_inputs,model, steps, thr):
    baseline = np.zeros(inputs.shape)
    # cu_inputs = torch.from_numpy(inputs).cuda()
    # cu_inputs.requires_grad_(True)

    # in_fea = model(cu_inputs)[1]
    # in_fea = in_fea.cpu().detach().numpy()
    # baseline = np.zeros(in_fea.shape)
    scaled_inputs = [inputs + (float(i) / steps) * ref + np.random.normal(size = inputs.shape, loc=0.0, scale=0.25) \
                    for i in range(-round(steps/2), round(steps/2) + 1)]
    scaled_inputs = np.asarray(scaled_inputs)
    
    scaled_inputs = torch.from_numpy(scaled_inputs)
    scaled_inputs = scaled_inputs.cuda()
    scaled_inputs.requires_grad_(True)
    # import ipdb
    # ipdb.set_trace()

    att_out, feas = model(scaled_inputs)

    # scaled_feas = np.asarray(feas.cpu().detach().numpy())
    
    score = torch.autograd.grad(att_out[:, label_inputs], feas, grad_outputs=torch.ones_like(att_out[:, label_inputs]), retain_graph=True)[0]
    # score = att_out[:, label_inputs]
    IA = torch.mean(score, dim=0)
    
    # model.zero_grad()
    # loss.backward()
    # grads = scaled_inputs.grad.data
    # avg_grads = torch.mean(grads, dim=0)
    # delta_X = scaled_inputs[-1] - scaled_inputs[0]

    delta_Y = feas[-1] - feas[0]
    integrated_grad = delta_Y * IA

    IG = integrated_grad.unsqueeze(0)
    tmp_sum = torch.sum(IG, (-1, -2))
    sort_sum, _ = torch.sort(abs(tmp_sum))
    T = sort_sum[0, -int(sort_sum.shape[1]*thr)]
    position = torch.where(abs(tmp_sum)>T)[1]
    loss = torch.sum(score * feas) + torch.sum(tmp_sum[0, position])


    model.zero_grad()
    loss.backward()
    grads = scaled_inputs.grad.data
    avg_grads = torch.mean(grads, dim=0)
    return avg_grads

def compare(a, b, c):
    signs = np.zeros_like(a)

    signs[a < 0] -= 1
    signs[a > 0] += 1

    signs[b < 0] -= 1
    signs[b > 0] += 1

    signs[c < 0] -= 1
    signs[c > 0] += 1

    return signs

# def NAA_random(cover, 
#         model, 
#         rho, 
#         payload, 
#         criterion = nn.CrossEntropyLoss(), 
#         max_beta = 1.0, 
#         beta_step = 0.1, 
#         wetCost = 1e13):
#     # 0 base, random
#     # init
#     # if rho is 'SUNIWARD' or rho is None:
#     #     rho = uniward(cover)
#     # elif rho is 'hill':
#     #     rho = hill(cover)
#     # else:
    
#     epsilon = 0.1
#     ens = 30

#     rhoP1 = np.reshape(rho[0], cover.size)
#     rhoM1 = np.reshape(rho[1], cover.size)


#     img_len = cover.size
#     msg_len = payload*img_len
#     # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
#     rand_ind = np.array(random.sample(range(img_len), k = img_len))
#     cover_1d = cover.flatten()

#     for beta in np.arange(0, max_beta + 0.01, beta_step):
#         # recover the stego for later usage
#         stego_1d = cover_1d.copy()

#         # divide
#         conv_ind = rand_ind[0 : int((1-beta)*img_len)]
#         adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
#         conv_cover = cover_1d[conv_ind]
#         adv_cover = cover_1d[adv_ind]
#         conv_rhoP1 = rhoP1[conv_ind].copy()
#         conv_rhoM1 = rhoM1[conv_ind].copy()
#         adv_rhoP1 = rhoP1[adv_ind].copy()
#         adv_rhoM1 = rhoM1[adv_ind].copy()
#         conv_rhoP1[conv_cover==255] = wetCost
#         conv_rhoM1[conv_cover==0] = wetCost
#         adv_rhoP1[adv_cover==255] = wetCost
#         adv_rhoM1[adv_cover==0] = wetCost

#         # embed conventional
#         if beta != 1.0:
#             conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
#             stego_1d[conv_ind] = conv_stego
        
#         # if beta == 0.0:
#         #     if isinstance(criterion, models.SiaStegNetLoss):
#         #         adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#         #                         torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#         #         if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#         #             attack_num = 1
#         #             return np.reshape(stego_1d, cover.shape), 1, attack_num
#         #     else:
#         #         adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#         #         if torch.argmax(model(adv_stego)[0],1) == 0:
#         #             attack_num = 1
#         #             return np.reshape(stego_1d, cover.shape), 1, attack_num
#         #     x_base = np.reshape(cover_1d.copy(), cover.shape)
#         #     continue

#         # calculate the gradients
#         model.zero_grad()
#         if isinstance(criterion, models.SiaStegNetLoss):
#             ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                          torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             ref_stego[0].requires_grad = True
#             ref_stego[1].requires_grad = True
#             cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
#             loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
#             loss.backward()
#             gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
#         else:
#             ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
#             ref_stego.requires_grad = True
#             #calculate the attribution
#             # IA = torch.zeros([1, 256, 16, 16]).cuda()  # SRNet
#             # IA = torch.zeros([1, 128, 32, 32]).cuda() # CovNet
#             IA = torch.zeros([1, 64, 64, 64]).cuda()  # CovNet group3
#             # IA = torch.zeros([1, 1, 256, 256]).cuda()
#             for m in range(ens):
#                 x_base = np.zeros_like(ref_stego.cpu().detach().numpy())
#                 cover_tmp = np.copy(ref_stego.cpu().detach().numpy())

#                 cover_tmp = cover_tmp*(m/ens) + np.random.uniform(-epsilon,epsilon,cover_tmp.shape)
#                 # cover_tmp = cover_tmp*(1-m/ens) + (m/ens)*x_base + np.random.uniform(-epsilon,epsilon,cover_tmp.shape)
#                 co = torch.from_numpy(cover_tmp).cuda()
#                 co.requires_grad = True


                
#                 output, feature = model(co)
#                 # import ipdb
#                 # ipdb.set_trace()
#                 # IA1 = IA1 + torch.autograd.grad(output, feature1, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#                 # IA2 = IA2 + torch.autograd.grad(output, feature2, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#                 IA = IA + torch.autograd.grad(output, feature, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#             IA = IA / ens
            
#             _, adv_feature = model(ref_stego)
#             _, base_feature = model(torch.from_numpy(np.reshape(x_base, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0))
#             attribution = (adv_feature - base_feature)*IA
#             # gradient = gradient.detach().cpu().numpy().flatten()[adv_ind]
#             # import ipdb
#             # ipdb.set_trace()

#             # gradient = torch.autograd.grad(criterion(model(ref_stego), torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
#             gradient = torch.autograd.grad(attribution, ref_stego, grad_outputs=torch.ones_like(attribution), retain_graph=True)[0]
#             gradient = gradient.detach().cpu().numpy().flatten()[adv_ind]

#         # adjust the distortion
#         adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
#         adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
#         adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
#         adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
#         adv_rhoP1[adv_rhoP1>wetCost] = wetCost
#         adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
#         adv_rhoM1[adv_rhoM1>wetCost] = wetCost
#         adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost

#         # x_base = np.reshape(stego_1d.copy(), cover.shape)

#         if beta != 0:
#             adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
#             stego_1d[adv_ind] = adv_stego.copy()

#         if isinstance(criterion, models.SiaStegNetLoss):
#             adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                               torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num
#         else:
#             adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#             if torch.argmax(model(adv_stego)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num

#     attack_num = round(beta*10)+1
#     conv_rhoP1 = rhoP1.copy()
#     conv_rhoM1 = rhoM1.copy()
#     conv_rhoP1[conv_cover==255] = wetCost
#     conv_rhoM1[conv_cover==0] = wetCost
#     conv_cover = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)
#     return conv_cover, 0, attack_num

# def NAA_coverbase_random(cover, 
#         model, 
#         rho, 
#         payload, 
#         criterion = nn.CrossEntropyLoss(), 
#         max_beta = 1.0, 
#         beta_step = 0.1, 
#         wetCost = 1e13):

#     # init
#     # if rho is 'SUNIWARD' or rho is None:
#     #     rho = uniward(cover)
#     # elif rho is 'hill':
#     #     rho = hill(cover)
#     # else:
    
#     epsilon = 0.1
#     ens = 30

#     rhoP1 = np.reshape(rho[0], cover.size)
#     rhoM1 = np.reshape(rho[1], cover.size)


#     img_len = cover.size
#     msg_len = payload*img_len
#     # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
#     rand_ind = np.array(random.sample(range(img_len), k = img_len))
#     cover_1d = cover.flatten()

#     for beta in np.arange(0, max_beta + 0.01, beta_step):
#         # recover the stego for later usage
#         stego_1d = cover_1d.copy()

#         # divide
#         conv_ind = rand_ind[0 : int((1-beta)*img_len)]
#         adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
#         conv_cover = cover_1d[conv_ind]
#         adv_cover = cover_1d[adv_ind]
#         conv_rhoP1 = rhoP1[conv_ind].copy()
#         conv_rhoM1 = rhoM1[conv_ind].copy()
#         adv_rhoP1 = rhoP1[adv_ind].copy()
#         adv_rhoM1 = rhoM1[adv_ind].copy()
#         conv_rhoP1[conv_cover==255] = wetCost
#         conv_rhoM1[conv_cover==0] = wetCost
#         adv_rhoP1[adv_cover==255] = wetCost
#         adv_rhoM1[adv_cover==0] = wetCost

#         # embed conventional
#         if beta != 1.0:
#             conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
#             stego_1d[conv_ind] = conv_stego
        
#         # if beta == 0.0:
#         #     if isinstance(criterion, models.SiaStegNetLoss):
#         #         adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#         #                         torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#         #         if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#         #             attack_num = 1
#         #             return np.reshape(stego_1d, cover.shape), 1, attack_num
#         #     else:
#         #         adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#         #         if torch.argmax(model(adv_stego)[0],1) == 0:
#         #             attack_num = 1
#         #             return np.reshape(stego_1d, cover.shape), 1, attack_num
#         #     x_base = np.reshape(cover_1d.copy(), cover.shape)
#         #     continue

#         # calculate the gradients
#         model.zero_grad()
#         if isinstance(criterion, models.SiaStegNetLoss):
#             ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                          torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             ref_stego[0].requires_grad = True
#             ref_stego[1].requires_grad = True
#             cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
#             loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
#             loss.backward()
#             gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
#         else:
#             ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
#             ref_stego.requires_grad = True
#             #calculate the attribution
#             # IA1 = torch.zeros([1, 16, 256, 256]).cuda()
#             # IA2 = torch.zeros([1, 16, 256, 256]).cuda()
#             # IA = torch.zeros([1, 256, 16, 16]).cuda()
#             IA = torch.zeros([1, 1, 256, 256]).cuda()
#             for m in range(ens):
#                 # x_base = np.zeros_like(ref_stego.cpu().detach().numpy())
#                 x_base = np.reshape(cover_1d.copy(), cover.shape)
#                 cover_tmp = np.copy(ref_stego.cpu().detach().numpy())

#                 cover_tmp = cover_tmp*(1-m/ens) + (m/ens)*x_base + np.random.uniform(-epsilon,epsilon,cover_tmp.shape)
#                 co = torch.from_numpy(cover_tmp).cuda()
#                 co.requires_grad = True

#                 # import ipdb
#                 # ipdb.set_trace()

#                 output, feature = model(co)
#                 loss = criterion(output, torch.tensor([0],dtype=torch.int64).cuda())
#                 loss.backward()
                
#                 # IA = IA + torch.autograd.grad(output, feature, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#                 IA = IA + co.grad.data
#             IA = IA / ens
            
#             _, adv_feature = model(ref_stego)
#             _, base_feature = model(torch.from_numpy(np.reshape(x_base, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0))
#             gradient = (adv_feature - base_feature)*IA
#             gradient = gradient.detach().cpu().numpy().flatten()[adv_ind]
            
#             # gradient = torch.autograd.grad(criterion(model(ref_stego), torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
#             # gradient = torch.autograd.grad(attribution, torch.tensor([0], dtype=torch.int64).cuda(), ref_stego)[0]

#         # adjust the distortion
#         adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
#         adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
#         adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
#         adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
#         adv_rhoP1[adv_rhoP1>wetCost] = wetCost
#         adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
#         adv_rhoM1[adv_rhoM1>wetCost] = wetCost
#         adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost

#         # x_base = np.reshape(stego_1d.copy(), cover.shape)

#         if beta != 0:
#             adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
#             stego_1d[adv_ind] = adv_stego.copy()

#         if isinstance(criterion, models.SiaStegNetLoss):
#             adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                               torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num
#         else:
#             adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#             if torch.argmax(model(adv_stego)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num

#     attack_num = round(beta*10)+1
#     conv_rhoP1 = rhoP1.copy()
#     conv_rhoM1 = rhoM1.copy()
#     conv_rhoP1[conv_cover==255] = wetCost
#     conv_rhoM1[conv_cover==0] = wetCost
#     conv_cover = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)
#     return conv_cover, 0, attack_num

# def NAA_stegobase_random(cover, 
#         model, 
#         rho, 
#         payload, 
#         criterion = nn.CrossEntropyLoss(), 
#         max_beta = 1.0, 
#         beta_step = 0.1, 
#         wetCost = 1e13):

#     # init
#     # if rho is 'SUNIWARD' or rho is None:
#     #     rho = uniward(cover)
#     # elif rho is 'hill':
#     #     rho = hill(cover)
#     # else:
    
#     epsilon = 0.1
#     ens = 30

#     rhoP1 = np.reshape(rho[0], cover.size)
#     rhoM1 = np.reshape(rho[1], cover.size)


#     img_len = cover.size
#     msg_len = payload*img_len
#     # rhoP1, rhoM1 = rho[:, :, 0].flatten(), rho[:, :, 0].flatten()
#     rand_ind = np.array(random.sample(range(img_len), k = img_len))
#     cover_1d = cover.flatten()

#     for beta in np.arange(0, max_beta + 0.01, beta_step):
#         # recover the stego for later usage
#         stego_1d = cover_1d.copy()

#         # divide
#         conv_ind = rand_ind[0 : int((1-beta)*img_len)]
#         adv_ind = rand_ind[int((1-beta)*img_len) : img_len]
#         conv_cover = cover_1d[conv_ind]
#         adv_cover = cover_1d[adv_ind]
#         conv_rhoP1 = rhoP1[conv_ind].copy()
#         conv_rhoM1 = rhoM1[conv_ind].copy()
#         adv_rhoP1 = rhoP1[adv_ind].copy()
#         adv_rhoM1 = rhoM1[adv_ind].copy()
#         conv_rhoP1[conv_cover==255] = wetCost
#         conv_rhoM1[conv_cover==0] = wetCost
#         adv_rhoP1[adv_cover==255] = wetCost
#         adv_rhoM1[adv_cover==0] = wetCost

#         # embed conventional
#         if beta != 1.0:
#             conv_stego = embd_sim(conv_cover, conv_rhoP1, conv_rhoM1, int(msg_len*(1-beta)))
#             stego_1d[conv_ind] = conv_stego
        
#         if beta == 0.0:
#             if isinstance(criterion, models.SiaStegNetLoss):
#                 adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                                 torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#                 if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#                     attack_num = 1
#                     return np.reshape(stego_1d, cover.shape), 1, attack_num
#             else:
#                 adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#                 if torch.argmax(model(adv_stego)[0],1) == 0:
#                     attack_num = 1
#                     return np.reshape(stego_1d, cover.shape), 1, attack_num
#             x_base = np.reshape(cover_1d.copy(), cover.shape)
#             continue

#         # calculate the gradients
#         model.zero_grad()
#         if isinstance(criterion, models.SiaStegNetLoss):
#             ref_stego = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                          torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             ref_stego[0].requires_grad = True
#             ref_stego[1].requires_grad = True
#             cls_logit, sub_fea1, sub_fea2 = model(*ref_stego)
#             loss = criterion(cls_logit, sub_fea1, sub_fea2, torch.tensor([0],dtype=torch.int64).cuda())
#             loss.backward()
#             gradient = torch.cat((ref_stego[0].grad.data, ref_stego[1].grad.data), -1)
#         else:
#             ref_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0) # (N,C,H,W)
#             ref_stego.requires_grad = True
#             #calculate the attribution
#             # IA1 = torch.zeros([1, 16, 256, 256]).cuda()
#             # IA2 = torch.zeros([1, 16, 256, 256]).cuda()
#             # IA = torch.zeros([1, 256, 16, 16]).cuda()
#             IA = torch.zeros([1, 1, 256, 256]).cuda()
#             for m in range(ens):
#                 # x_base = np.zeros_like(ref_stego.cpu().detach().numpy())
#                 cover_tmp = np.copy(ref_stego.cpu().detach().numpy())

#                 cover_tmp = cover_tmp*(1-m/ens) + (m/ens)*x_base + np.random.uniform(-epsilon,epsilon,cover_tmp.shape)
#                 co = torch.from_numpy(cover_tmp).cuda()
#                 co.requires_grad = True

#                 # import ipdb
#                 # ipdb.set_trace()
                
#                 output, feature = model(co)
#                 # IA1 = IA1 + torch.autograd.grad(output, feature1, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#                 # IA2 = IA2 + torch.autograd.grad(output, feature2, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#                 IA = IA + torch.autograd.grad(output, feature, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
#             IA = IA / ens
            
#             _, adv_feature = model(ref_stego)
#             _, base_feature = model(torch.from_numpy(np.reshape(x_base, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0))
#             gradient = (adv_feature - base_feature)*IA
#             gradient = gradient.detach().cpu().numpy().flatten()[adv_ind]
            
#             # gradient = torch.autograd.grad(criterion(model(ref_stego), torch.tensor([0],dtype=torch.int64).cuda()), ref_stego)[0]
#             # gradient = torch.autograd.grad(attribution, torch.tensor([0], dtype=torch.int64).cuda(), ref_stego)[0]

#         # adjust the distortion
#         adv_rhoP1[gradient>0] = adv_rhoP1[gradient>0]*2
#         adv_rhoP1[gradient<0] = adv_rhoP1[gradient<0]/2
#         adv_rhoM1[gradient<0] = adv_rhoM1[gradient<0]*2
#         adv_rhoM1[gradient>0] = adv_rhoM1[gradient>0]/2
#         adv_rhoP1[adv_rhoP1>wetCost] = wetCost
#         adv_rhoP1[np.isnan(adv_rhoP1)] = wetCost
#         adv_rhoM1[adv_rhoM1>wetCost] = wetCost
#         adv_rhoM1[np.isnan(adv_rhoM1)] = wetCost

#         x_base = np.reshape(stego_1d.copy(), cover.shape)

#         if beta != 0:
#             adv_stego = embd_sim(adv_cover, adv_rhoP1, adv_rhoM1, int(msg_len*beta))
#             stego_1d[adv_ind] = adv_stego.copy()

#         if isinstance(criterion, models.SiaStegNetLoss):
#             adv_stego_eval = [torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,:cover.shape[1]//2]).float().cuda().unsqueeze(0).unsqueeze(0),
#                               torch.from_numpy(np.reshape(stego_1d, cover.shape)[:,cover.shape[1]//2:]).float().cuda().unsqueeze(0).unsqueeze(0)]
#             if torch.argmax(model(*adv_stego_eval)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num
#         else:
#             adv_stego = torch.from_numpy(np.reshape(stego_1d, cover.shape)).float().cuda().unsqueeze(0).unsqueeze(0)
#             if torch.argmax(model(adv_stego)[0],1) == 0:
#                 attack_num = round(beta*10)+1
#                 return np.reshape(stego_1d, cover.shape), 1, attack_num

#     attack_num = round(beta*10)+1
#     conv_rhoP1 = rhoP1.copy()
#     conv_rhoM1 = rhoM1.copy()
#     conv_rhoP1[conv_cover==255] = wetCost
#     conv_rhoM1[conv_cover==0] = wetCost
#     conv_cover = embd_sim(cover_1d, conv_rhoP1.flatten(), conv_rhoM1.flatten(), msg_len).reshape(cover.shape)
#     return conv_cover, 0, attack_num