import torch
import torch.nn.functional as F
import numpy as np

def cosine_matrix(M, N):
    cosine_matrix = torch.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if i == 0:
                cosine_matrix[i, j] = np.sqrt(1 / N)
            else:
                cosine_matrix[i, j] = np.sqrt(2 / N) * np.cos((np.pi * i * (1 / 2 + j)) / N)
    return cosine_matrix

def normal(value):
    return (value - np.min(value)) / (np.max(value) - np.min(value))

def dct2(img, kernel_size):
    batch, channels, M, N = img.shape
    output = torch.zeros_like(img)
    
    C = cosine_matrix(kernel_size, kernel_size).to(img.device)
    
    if kernel_size < M:
        for i in range(int(N / kernel_size)):
            for j in range(int(M / kernel_size)):
                tmp = img[:, :, j*8:(j+1) * 8, i*8:(i+1)*8]
                output[:, :, j*8:(j+1) * 8, i*8:(i+1)*8] = torch.matmul(C, torch.matmul(tmp, C.T))
            
    #print(output)
        
    return output

def idct2(img, kernel_size):
    batch, channels, M, N = img.shape
    output = torch.zeros_like(img)
    
    C = cosine_matrix(kernel_size, kernel_size).to(img.device)
    
    if kernel_size < M:
        for i in range(int(N / kernel_size)):
            for j in range(int(M / kernel_size)):
                tmp = img[:, :, j*8:(j+1) * 8, i*8:(i+1)*8]
                output[:, :, j*8:(j+1) * 8, i*8:(i+1)*8] = torch.matmul(C, torch.matmul(tmp, C.T))
            
    #print(output)
        
    return output

def dct2_ana(img, N, M = None):
    
    M = N if M == None else M
    
    C = cosine_matrix(M, N).to(img.device)
    C = C.unsqueeze(0).unsqueeze(0)
    weight = torch.cat([C, C, C])
    #weight = weight.unsqueeze(0)
    #weight = torch.cat([weight, weight, weight])
    
    output = F.conv2d(img, weight, groups = 3)
    print(np.log2(N))
        
    return torch.clamp(output / np.log2(N), 0, 1)

            