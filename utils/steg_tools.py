from scipy import misc, io, signal
import cv2
import math, sys
import os
import numpy as np
from scipy.signal import convolve2d
import random

def embd_sim(cover,
             rhoP1,
             rhoM1,
             m,
             seed = None): # m is the bits that should be embedded

    n = rhoP1.size

    # calc lambda
    Lambda = calc_lambda(rhoP1,rhoM1,m,n)

    # calc change rate
    pChangeP1 = (np.exp(-Lambda*rhoP1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
    pChangeM1 = (np.exp(-Lambda*rhoM1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))

    # make modification
    if seed is not None:
        np.random.seed(seed)
    randChange = np.random.rand(np.size(rhoP1))
    modification = np.zeros(np.size(rhoP1))
    modification[randChange<pChangeP1]=1
    modification[randChange>=1-pChangeM1]=-1
    stego=cover+modification

    # correction
    stego[stego>255]=255
    stego[stego<0]=0
    return stego

def calc_lambda(rhoP1,rhoM1,message_length,n):
    l3 = 1e+3
    m3 = message_length+1
    iterations = 0
    while m3 > message_length:
          l3 = l3*2
          pP1 = (np.exp(-l3*rhoP1))/(1+np.exp(-l3*rhoP1)+np.exp(-l3*rhoM1))
          pM1 = (np.exp(-l3*rhoM1))/(1+np.exp(-l3*rhoP1)+np.exp(-l3*rhoM1))
          m3 = ternary_entropyf(pP1,pM1)
          iterations = iterations+1
          if iterations>10:
             Lambda = l3
             return Lambda

    l1 = 0
    m1 = n
    Lambda = 0
   
    alpha = float(message_length)/n
    while (float(m1-m3)/n>alpha/1000.0) and (iterations<30):
             Lambda = l1+(l3-l1)/2.0
             pP1 = (np.exp(-Lambda*rhoP1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
             pM1 = (np.exp(-Lambda*rhoM1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
             m2 = ternary_entropyf(pP1,pM1)
             if m2<message_length:
                l3 = Lambda
                m3 = m2
             else:
                l1 = Lambda
                m1 = m2
             iterations = iterations+1
    return Lambda

def ternary_entropyf(pP1,pM1):
    p0=1-pP1-pM1
    p0[p0==0] = 1e-10
    pP1[pP1==0] = 1e-10
    pM1[pM1==0] = 1e-10 
    Ht = -pP1*np.log2(pP1)-pM1*np.log2(pM1)-(p0)*np.log2(p0)
    Ht = np.sum(Ht)
    return Ht





def uniward(img):
    sgm = 1
    hpdf = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539,\
            -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
    lpdf = np.multiply(((-1)**np.array(range(0, len(hpdf)))), np.flipud(hpdf))
    hpdf.shape = (1, len(hpdf))
    lpdf.shape = (1, len(lpdf))
    F1 = np.matmul(np.transpose(lpdf), hpdf)
    F2 = np.matmul(np.transpose(hpdf), lpdf)
    F3 = np.matmul(np.transpose(hpdf), hpdf)
    F = (F1, F2, F3)
    img = img.astype(float)
    p = -1
    wetCost = 1e13
    padSize = max([len(F1), len(F1[0]), len(F2), len(F2[0]), len(F3), len(F3[0])])
    coverPadded = np.lib.pad(img, padSize, 'symmetric')
    xii = []
    for fIndex in range(0, 3):
        R = np.rot90(convolve2d(np.rot90(coverPadded, 2), np.rot90(F[fIndex], 2), mode='same'), 2)
        xi = np.rot90(convolve2d(np.rot90(1/(abs(R)+sgm), 2), np.rot90(np.rot90(abs(F[fIndex]), 2), 2), mode='same'), 2)
        if len(F[fIndex]) % 2 == 0:
            xi = np.roll(xi, 1, axis=0)
        if len(F[fIndex][0]) % 2 == 0:
            xi = np.roll(xi, 1, axis=1)
        xi = xi[int((len(xi)-len(img))/2):int(len(xi)-(len(xi)-len(img))/2), int((len(xi[0])-len(img[0]))/2):int(len(xi[0])-(len(xi[0])-len(img[0]))/2)]
        xii.append(xi)
    rho = xii[0]+xii[1]+xii[2]
    rho[rho > wetCost] = wetCost
    rho[np.isnan(rho)] = wetCost
    rhoP1 = rho
    rhoM1 = rho
    rhoP1[img == 255] = wetCost
    rhoM1[img == 0] = wetCost
    return np.stack((rhoP1, rhoM1), 2)


def hill(img):
    # initialization

    wetCost = 1e13
    wetThre = 1e3
    F = np.array([[-0.25, 0.5, -0.25],
        [0.5,    -1,    0.5],
        [-0.25, 0.5, -0.25]])

    # compute residual
    R = cv2.filter2D(img, -1, F, borderType=cv2.BORDER_REFLECT)
    # compute suitability
    xi = cv2.filter2D(abs(R), -1, np.array([[1 for col in range(3)] for row in range(3)])/9.0, borderType=cv2.BORDER_REFLECT)
    # compute embedding cost \rho
    with np.errstate(divide='ignore'):
        xi2 = 1.0/xi

    xi2[xi2 > wetCost] = wetCost
    rho = cv2.filter2D(xi2, -1, 1/225.0*np.array([[1 for col in range(15)] for row in range(15)]), borderType = cv2.BORDER_REFLECT)

    rho[rho > wetThre] = wetCost
    rho[np.isnan(rho)] = wetCost

    rhoP1 = rho
    rhoM1 = rho

    rhoP1[img == 255] = wetCost
    rhoM1[img == 0] = wetCost
    return np.stack((rhoP1, rhoM1), 0)