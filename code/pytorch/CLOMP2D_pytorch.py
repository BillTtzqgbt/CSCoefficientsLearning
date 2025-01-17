# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:23:49 2024

@author: BillT
"""


#GlobalPath = 'D:\Code'
#DataPath = 'home/BillT/Document/Data/'
# import paddle
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
import pywt

import time
import cv2

# OpenMP
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.cuda.set_device('cuda:0')

img1 = cv2.imread("/home/Data/1_1920x1080.tif",cv2.IMREAD_UNCHANGED)
print(np.shape(img1))

input_img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)  # to gray

H, W = np.shape(input_img)

# ---normalise to 0 - 1
img = input_img.astype('float32')
img = img/np.max(img)  # [0,1]

n = H  #number of sampling points
r = 0.3
m = np.int32(n*r)

def GenerateMeaMtx(dataN, SampPer, Type='Gaus'):
    dataM = np.int32(dataN*SampPer)
    # -----Bernoulli
    if Type=='Bernoulli':
        MeaMtx = np.zeros([dataM, dataN])
        a1 = np.arange(dataN)
        a2 = np.arange(dataM)
        b1 = np.random.permutation(a1)
        b2 = np.random.permutation(a2)

        MeaMtx[b2, b1[0:dataM]] = 1.0
    elif Type =='Gaus':
        MeaMtx = np.random.randn(m,n)
        # CNorm = 1.0/np.sqrt(np.linalg.norm(MeaMtx,axis=0))
        CNorm = 1.0/np.linalg.norm(MeaMtx,axis=0)
        MeaMtx = MeaMtx*CNorm
        

    return MeaMtx

import scipy.fftpack as fft


Mtx = GenerateMeaMtx(n,r,'Gaus')
Y = np.matmul(Mtx, img)


d1 = np.diag(np.ones([n]))
DCTBase = fft.dct(d1, axis=0, norm='ortho').transpose()#no transpose for MRI image

D = DCTBase
A=np.matmul(Mtx, D)
ANormSeed = 1.0/np.linalg.norm(A,axis=0)

A = A*ANormSeed

AT = torch.tensor(A[:, :n], dtype=torch.float32)
yT = torch.tensor(Y, dtype=torch.float32).permute(1, 0)
DT = torch.tensor(D, dtype=torch.float32).permute(1, 0)
TimeAll = np.zeros([3])
#%%

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class CSRec1(nn.Module):
    def __init__(self, A, y, neurons=32):
        super(CSRec1, self).__init__()
        self.m, self.n = np.shape(A)
        
        self.A = A
        self.y = y
        self.neurons = neurons
        self.fc1 = nn.Linear(in_features=self.m, out_features=self.neurons)#out_features=self.neurons)
        # self.fc2 = nn.Linear(in_features=self.neurons, out_features=self.neurons)
        # self.fc3 = nn.Linear(in_features=self.neurons, out_features=self.neurons)
        self.fc4 = nn.Linear(in_features=self.neurons, out_features=self.n)

        
    def forward(self,res, WSup, Th=0.7):
        h = F.relu(self.fc1(self.y))#s, sparse coefficients for just 1 neural layer
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)#s, sparse coefficients
        
        corr = torch.abs(torch.matmul(res, self.A))
        corr = corr / torch.max(corr, dim=1, keepdim=True)[0]
        Mask = (corr>Th)

        Mask = torch.logical_or(Mask, WSup.bool())

        h0 = h*Mask.to(dtype=torch.float32)
        
        y_pred = torch.matmul(h0, self.A.permute(1, 0))
        
        return Mask, y_pred, h
    
modelTest = CSRec1(AT, yT, neurons=16)

res = torch.randn(W, m)
WSup = torch.randn(W, n)
summary(modelTest, input_data=[res, WSup], dtypes=[torch.float32])


class CSRec2(nn.Module):
    def __init__(self, A, W):
        super(CSRec2, self).__init__()
        self.m, self.n = np.shape(A)
        self.weight = nn.Parameter(torch.empty((W, self.n), dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)
        self.A = torch.tensor(A, dtype=torch.float32)

    def forward(self, res, WSup, Th=0.7):
        
        corr = torch.abs(torch.matmul(res, self.A))
        corr = corr / torch.max(corr, dim=1, keepdim=True)[0]
        Mask = (corr>Th)

        Mask = torch.logical_or(Mask, WSup.bool())
        
        s = self.weight * Mask.float()
        
        y_pred = torch.matmul(self.A, s.permute(1, 0)).permute(1, 0)

        return Mask, y_pred, self.weight
    


modelTest = CSRec2(A,2)

res = torch.randn(2, m)
WSup = torch.randn(2, n)
summary(modelTest, input_data=[res, WSup], dtypes=[torch.float32])

class CSRec3(nn.Module):
    def __init__(self, A, y, neurons=32):
        super(CSRec3, self).__init__()
        self.m, self.n = np.shape(A)
        
        self.A = A
        self.y = y
        self.neurons = neurons
        self.fc1 = nn.Linear(in_features=self.m, out_features=self.neurons)#out_features=self.neurons)
        # self.fc2 = nn.Linear(in_features=self.neurons, out_features=self.neurons)
        # self.fc3 = nn.Linear(in_features=self.neurons, out_features=self.neurons)
        self.fc4 = nn.Linear(in_features=self.neurons, out_features=self.n)
        
        #--------method 2, full neural connection rather than matrix multiply
        self.fcRes = nn.Linear(in_features=self.m, out_features=self.n, bias=False)
        self.fcRes.weight.data = self.A.clone()
        self.fcRes.weight.requires_grad_(False)
        
        self.fcLast = nn.Linear(in_features=self.n, out_features=self.m, bias=False)
        self.fcLast.weight.data = self.A.t().clone()
        self.fcLast.weight.requires_grad_(False)

        
    def forward(self,res, WSup, Th=0.7):
        h = F.relu(self.fc1(self.y))#s, sparse coefficients for just 1 neural layer
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)#s, sparse coefficients

        corr = torch.abs(self.fcLast(res))
        corr = corr / torch.max(corr, dim=1, keepdim=True)[0]
        Mask = (corr>Th)
        Mask = torch.logical_or(Mask, WSup.bool())

        h0 = h * Mask.to(dtype=torch.float32)
        
        y_pred = self.fcRes(h0)
        
        
        return Mask, y_pred, h
    
modelTest = CSRec3(AT, yT, neurons=16)

res = torch.randn(W, m)
WSup = torch.randn(W, n)
summary(modelTest, input_data=[res, WSup], dtypes=[torch.float32])

#%%
def train(net, optimizer, yIn, epoch_num = 10):
    lossAll = []
    NewS = []

    print('Start training...')
    residual = yIn
    net.train()

    WSup = torch.zeros((1, n), dtype=torch.bool)

    #
    ThNew=0.7

    for epoch in range(epoch_num):
        WSup, y_pred, s = net(residual,WSup,ThNew)
        
        residual = yIn - y_pred

        NewS.append(torch.sum(WSup).detach().numpy())

        loss1 = F.mse_loss(y_pred, yIn,reduction='mean')
        
        imgPred = torch.matmul(s, DT)
        
        Var = (torch.pow(imgPred[1:, :] - imgPred[:W - 1, :], 2).mean()) + \
              (torch.pow(imgPred[:, 1:] - imgPred[:, :H - 1], 2).mean())
                   

        ImgBlock = F.unfold(imgPred.unsqueeze(0).unsqueeze(0), kernel_size=(3, 3), stride=2, padding=0)
        loss2 = torch.mean(torch.std(ImgBlock, dim=1))
        
        loss = loss1 + 0.1*loss2 +0.2*Var#
        
        #---------------step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # lossNP = loss.numpy()
        lossNP = loss.detach().numpy()
        lossAll.append(lossNP)
        Lr.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(0.1, param_group['lr'])
        
        print('epoch = %d    loss = %.6f   lr = %.5f' % (epoch, lossNP, optimizer.param_groups[0]['lr']))
        #####----------------------------------
        if epoch>80:
            if (lossNP<1e-4)|(epoch==epoch_num-1)|(np.abs(lossAll[-2]-lossAll[-1])<1e-7):#quick visualisation
                break
        
    return WSup, lossAll,s,NewS

#---------------------------------

Ts = time.time()

# model = CSRec1(AT,yT,neurons=128)#try 0.1*Var + 0.2*loss2, learning rate =0.01 to 0.001
# model = CSRec3(AT,yT,neurons=128)#try 0.1*Var + 0.2*loss2, learning rate =0.01 to 0.001
model = CSRec2(A=AT,W=W)#try 0.1*Var + 0.2*loss2, learning rate =2 to 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=2)
Lr = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=100)  # 

Enum = 150
Sup,lossAll,sPred,NewS =train(net=model, optimizer=optimizer, yIn=yT,epoch_num=Enum)

TimeAll[0]=time.time()-Ts
print('Time: ',TimeAll[0])

plt.figure()
plt.semilogy(lossAll,'--r')

plt.title('loss curve')
plt.grid(visible=True)
plt.show()

x_pred = np.matmul(D, np.transpose(sPred.detach().numpy()))
imgPred = x_pred

imgPred[imgPred<0]=0
imgPred[imgPred>1]=1


plt.figure()
plt.subplot(1,2,1)
plt.imshow(input_img)
plt.subplot(1,2,2)
plt.imshow(imgPred)
plt.show()

print('SSIM: ',ssim(img,imgPred,data_range=img.max()-img.min()))

#%%  OMP in GPU
def OMP1DP(y,A,Th):
    r = y.clone()
    
    lossAll = []
    n = np.shape(A)[1]

    Sup = torch.zeros((W, n), dtype=torch.bool)
    WAll = torch.zeros((W, n))

    for ii in range(200):
       print(ii)

       Z = torch.abs(torch.matmul(r, A))
       Z = Z / (torch.max(Z, dim=1, keepdim=True)[0] + 1e-6)
       
       pos = (Z>0.7)

       Sup = torch.logical_or(pos, Sup)

       for jj in range(W):

           ZeroDet = torch.sum(Sup[jj, :])
           
           if ZeroDet!=0:
               weight = torch.matmul(torch.pinverse(A[:, Sup[jj, :]]), y[jj, :])
               r[jj, :] = y[jj, :] - torch.matmul(A[:, Sup[jj, :]], weight)
               WAll[jj,Sup[jj,:]]=weight

       Loss = torch.mean(r.pow(2))

       lossAll.append(Loss.detach().numpy())
       if Loss<Th:
           
           break
    
    return WAll, lossAll

Ts = time.time()

WeiAll, lossOMPp = OMP1DP(yT,AT,Th=1e-4)

sRecP = WeiAll.detach().numpy()
sRecP = sRecP.transpose()

TimeAll[1]=time.time()-Ts
print(TimeAll[1])


xOMP = np.matmul(D, sRecP)
print('SSIM: ',ssim(img,xOMP,data_range=img.max()-img.min()))

#%% IHT on GPU
def IHTp(y,A,Th):
    r = y.clone()

    u = 0.1
    lossIHT = []
    W, m = np.shape(y)
    K = np.int32(m/5)
    
    s = torch.zeros((W, n), dtype=torch.float32)
    for ii in range(200):
       print(ii)
       tmp = torch.matmul(r, A)

       s = s + u*tmp

       AbsTmp = torch.abs(s)
       _, Idx = torch.topk(AbsTmp, K, dim=-1, largest=True, sorted=False)

       mask = torch.zeros_like(s, dtype=torch.bool)
       mask.scatter_(1, Idx, True)
       s = torch.where(mask, s, torch.tensor(0.0))

       r = y - torch.matmul(s, A.permute(1, 0))

       Loss = torch.mean(r.pow(2))
      
       lossIHT.append(Loss.detach().numpy())

       if Loss<Th:
           break
    
    return s, lossIHT

Ts = time.time()


sIHTp, lossIHT = IHTp(yT,AT,Th=1e-4)
sIHT = sIHTp.numpy().transpose()
    
    

TimeAll[2]=time.time()-Ts
print(TimeAll[2])


xIHT = np.matmul(D, sIHT)

print(ssim(img,xIHT,data_range=img.max()-img.min()))

#
#%% OMP on CPU
def OMP1D(y,A,Th):
    r = y.copy()
    Sup = np.zeros([W,n],dtype='bool')
    WAll = np.zeros([W,n],dtype='float32')

    lossAll = []
    for ii in range(200):
       print(ii)
       Z = np.abs(np.matmul(r,A))
       

       Z = Z/(np.max(Z,axis=1,keepdims=True) + 1e-6)
 
       pos = (Z>0.7)

       Sup = np.bitwise_or(pos, Sup)
       
       for jj in range(W):
 
           ZeroDet = np.sum(Sup[jj,:])
           
           
           if ZeroDet!=0:
               pos1 = Sup[jj,:]
               weight = np.matmul(np.linalg.pinv(A[:,pos1]), y[jj,:])
               r[jj,:] = y[jj,:] - np.matmul(A[:,pos1],weight)

               WAll[jj,pos1]=weight

       Loss = np.mean(r**2)
       
       
       lossAll.append(Loss)
       if Loss<Th:
           break
    
    return WAll, lossAll

Ts = time.time()



sRec, lossOMP = OMP1D(Y.transpose().astype('float32'),A.astype('float32'),Th=1e-4)


TimeAll[1]=time.time()-Ts
print(time.time()-Ts)

xOMP = np.matmul(D, sRec.transpose())
print(ssim(img,xOMP,data_range=img.max()-img.min()))

#%% IHT on CPU
def IHT(y,A,Th):
    r = y.copy()

    u = 0.1
    lossIHT = []
    W, m = np.shape(y)
    K = np.int32(m/5)
    
    s = np.zeros([W,n],dtype='float32')

    for ii in range(200):
       print(ii)
       tmp = np.matmul(r,A)

       s = s + u*tmp

       AbsTmp = np.abs(s)
       Idx = np.argsort(AbsTmp,axis=-1)
       Idxno = Idx[:,0:n-K]
       
       for jj in range(W):
           s[jj,Idxno[jj,:]]=0.0
           

       r = y - np.matmul(s,A.transpose())

       Loss = np.mean(r**2)
      
       lossIHT.append(Loss)
       
       if Loss<Th:
           break
    
    return s, lossIHT

Ts = time.time()


sIHT, lossIHT = IHT(Y.transpose().astype('float32'),A.astype('float32'),Th=1e-4)
    

print(time.time()-Ts)
# print(np.shape(SupPos))

xIHT = np.matmul(D, sIHT.transpose())

print(ssim(img,xIHT,data_range=img.max()-img.min()))

#%%

x_pred = np.matmul(D, np.transpose(sPred.detach().numpy()))

# x_pred = np.matmul(D, np.transpose(sPred.numpy()))


plt.figure()
plt.subplot(2,2,1)
plt.imshow(input_img)
plt.title('Ground truth')
plt.subplot(2,2,2)
imgPred = x_pred
imgOMP = xOMP
imgIHT = xIHT

imgOMP[imgOMP<0]=0# remove outlier
imgOMP[imgOMP>1]=1

imgPred[imgPred<0]=0
imgPred[imgPred>1]=1

imgIHT[imgIHT<0]=0
imgIHT[imgIHT>1]=1

print('--Time:[CLOMP, OMP, IHT]----')
print(TimeAll)

print('--SSIM----')
print(ssim(img,imgPred,data_range=img.max()-img.min()))
print(ssim(img,imgOMP,data_range=img.max()-img.min()))
print(ssim(img,imgIHT,data_range=img.max()-img.min()))

print('--MSE----')
print(np.mean((img-imgPred)**2.0))
print(np.mean((img-imgOMP)**2.0))
print(np.mean((img-imgIHT)**2.0))


plt.imshow(imgPred)
plt.title('CLOMP')
plt.subplot(2,2,3)


plt.imshow(imgOMP)
plt.title('OMP')
plt.subplot(2,2,4)
plt.imshow(imgIHT)
plt.title('IHT')
plt.show()