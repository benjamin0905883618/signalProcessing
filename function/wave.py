import pywt
import torch
import numpy as np
import matplotlib.pyplot as plt

def normal(value):
    return (value - np.min(value)) / (np.max(value) - np.min(value))

def discrete_wavelet(imgs, wave, level = 1, cuda = True):
    LL = imgs.detach()
    for i in range(level):
        LL, (LH, HL, HH) = pywt.dwt2(LL, wave)
    result = LL
    for i in range(level):
        result = pywt.idwt2((result, (torch.zeros(result.shape), torch.zeros(result.shape), torch.zeros(result.shape))), wave)
    
    result, LL, LH, HL, HH = torch.FloatTensor(normal(result)), torch.FloatTensor(normal(LL)), torch.FloatTensor(normal(LH)), torch.FloatTensor(normal(HL)), torch.FloatTensor(normal(HH))
    if cuda:
        result, LL, LH, HL, HH = result.cuda(), LL.cuda(), LH.cuda(), HL.cuda(), HH.cuda()
    
    return result, (LL, LH, HL, HH)

def continuous_wavelet(imgs, wave, level = 1, cuda = True):
    LL = imgs.numpy()
    coef, freqs = pywt.cwt(LL, np.arange(1, level+1), wave, method = 'conv')
    #print(f'the coef = {coef.shape}')
    #print(f'the freqs = {freqs}')
    for i in range(coef.shape[0]):
        #print(np.max(coef[i]))
        coef[i] = normal(coef[i])
        print(f'minimum = {np.min(coef[i])}, maximum = {np.max(coef[i])}')
    output = torch.from_numpy(coef)
    return output.cuda() if cuda else output 