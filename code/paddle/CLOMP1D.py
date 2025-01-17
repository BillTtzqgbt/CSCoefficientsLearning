
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 W:28:45 2024

@author: BillTang
"""

import paddle
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
paddle.set_device('gpu:2')
import scipy.fftpack as fft

n = 1000  #number of sampling points

d1 = np.diag(np.ones([n]))

DCTBase = fft.dct(d1, axis=0, norm='ortho').transpose()

D = DCTBase

#%% prepare basic functions

def Gen1SparseFun(K,n):
    # generate K-sparse signal
    #K is sparse number
    
    s = np.zeros(n)
    
    s0 = np.linspace(-10,5,K)
    Svalue = 0.5**s0+25
    Svalue = Svalue/np.max(Svalue)
    
    a1 = np.arange(np.int32(np.max([K, 0.9*n])))#cut off high frequency, must>K
    a2 = np.arange(K)
    a3 = np.arange(np.int32(n/10))
    b1 = np.random.permutation(a1)
    b2 = np.random.permutation(a2)
    b3 = np.random.permutation(a3)
    
    Svalue[b2[0:np.int32(K/5)]] = -Svalue[b2[0:np.int32(K/5)]]
    s[b1[0:K]] = Svalue
    s[0] = 5
    s[b3[0:4]] =5*np.array([0.8,0.75,0.6,0.5])
    s[np.int32(np.max([K, 0.9*n])):]=0.01*np.random.rand(n-np.int32(np.max([K, 0.9*n])))
    
    x = fft.idct(s)
    
    Max=np.max(x)#normalise to 0 ~1,
    Min=np.min(x)
    x = x/(Max-Min)-Min/(Max-Min)

    return s,x


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

def ssim1D(x1, x2, winS=11):

  x1 = x1.astype(np.float64)
  x2 = x2.astype(np.float64)
  kernel = cv2.getGaussianKernel(winS, 1.5)
  window = kernel#np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(x1, -1, window)#[1:-1, 1:-1] # valid
  mu2 = cv2.filter2D(x2, -1, window)#[1:-1, 1:-1]
  
  DR1 = np.max(x1,axis=0,keepdims=True)
  DR2 = np.min(x1,axis=0,keepdims=True)
  DR3 = np.matmul(np.ones([np.shape(mu1)[0], 1]),DR1-DR2)
  DR = np.reshape(DR3,[np.shape(mu1)[0], -1])
  
  C1 = (0.06 * DR)**2
  C2 = (0.1 * DR)**2
  
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu12 = mu1 * mu2
  sigma1_sq = cv2.filter2D(x1**2, -1, window) - mu1_sq#[5:-5, 5:-5]
  sigma2_sq = cv2.filter2D(x2**2, -1, window) - mu2_sq
  sigma12 = cv2.filter2D(x1 * x2, -1, window) - mu12
  # ssim_map =  (2.0*sigma12 + C1)/(sigma1_sq + sigma2_sq + C2)
  
  ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  # print(np.shape(ssim_map))
  ssim = np.mean(ssim_map,axis=0)
  return ssim

# %% CS measurement
r = 0.15 #measurement rate
m = np.int32(n*r)
Sr = 0.4#sparse rate
W=1000
RFun = np.zeros([n,W])


for ii in range(W):
    s0,RFun[:,ii] = Gen1SparseFun(np.int32(Sr*n),n)#adjust 

#---------------------------------------------------------------------
plt.plot(s0)
plt.show()

Mtx = GenerateMeaMtx(n,r,'Gaus')

y = np.matmul(Mtx, RFun)


A=np.matmul(Mtx, D)
ANormSeed = 1.0/np.linalg.norm(A,axis=0)
A = A*ANormSeed

AT = paddle.to_tensor(A).astype('float32')
yT = paddle.to_tensor(y).astype('float32').transpose(perm=[1,0])
DT = paddle.to_tensor(D).astype('float32').transpose(perm=[1,0])


TimeAll = np.zeros([3])
#%%
import paddle.nn as nn
import paddle.nn.functional as F
# paddle.set_device('gpu:2')
    

class CSRec2(nn.Layer):
    def __init__(self, A, W):
        super(CSRec2, self).__init__()
        self.m, self.n = np.shape(A)
        nn.initializer.set_global_initializer(nn.initializer.Constant(0.2), nn.initializer.Constant(0.2))
        self.weight = self.create_parameter([W,self.n],dtype='float32')
        self.A = paddle.to_tensor(A).astype('float32')

    def forward(self, res, WSup, Th=0.7):

        corr = paddle.abs(paddle.matmul(res, self.A))
        corr = corr/paddle.max(corr,axis=1,keepdim=True)
        Mask = (corr>Th)

        Mask = paddle.bitwise_or(Mask, WSup.astype('bool'))
        
        s = self.weight*Mask.astype('float32')

        y_pred = paddle.matmul(self.A, s,transpose_y=True).transpose([1,0])
        
        return Mask, y_pred, self.weight
    

modelTest = CSRec2(A,2)
paddle.summary(modelTest,input_size=[(2,m),(2,n)],dtypes='float32')

#%%
StopTh = 1e-5


def train(net, optimizer, yIn, epoch_num = 10):
    lossAll = []

    print('Start training...')
    residual = yIn
    net.train()

    WSup = paddle.zeros((1,n),dtype='bool')
    
    ThNew=0.7

    for epoch in range(epoch_num):
        WSup, y_pred, s = net(residual,WSup,ThNew)
        
        residual = yIn - y_pred
        
        loss1 = F.mse_loss(y_pred, yIn,reduction='mean')

        
        xPred = paddle.matmul(s,DT)

        Var =  F.mse_loss(xPred[:,1:], xPred[:,:(n-1)],reduction='mean')#paddle.pow(xPred[0,1:] - xPred[0,:(n-1)], 2).mean() 
        
        loss = loss1 +0.5*Var

        #-------------step
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        lossNP = loss.numpy()
        lossAll.append(lossNP)
        Lr.step()

        print('epoch = %d    loss = %.6f   lr = %.5f' % (epoch, lossNP, optimizer.get_lr()))
        #####----------------------------------
        if epoch>80:
            if (loss1.numpy()<StopTh)|(epoch==epoch_num-1):#|(np.abs(lossAll[-2]-lossAll[-1])<1e-6):#quick visualisation
                
                break
        
    return WSup, lossAll,s

#--------------------------------
Ts = time.time()

EpNum = 200
model = CSRec2(AT,W)

Lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=2,decay_steps=190,end_lr=0.01)

Opt = paddle.optimizer.Adam(learning_rate=Lr, parameters=model.parameters())


Sup,lossAll,sPred =train(net=model, optimizer=Opt, yIn=yT,epoch_num=EpNum)

TimeAll[0]=time.time()-Ts
print('Time cost: ',time.time()-Ts)
xPred = paddle.matmul(sPred,DT)
xPred = xPred.transpose(perm=[1,0])
xPred[xPred<-1]=-1
xPred[xPred>1]=1
print('SSIM: ',np.mean(ssim1D(np.squeeze(RFun),np.squeeze(xPred.numpy()))))



#%%  OMP in paddle, GPU
def OMP1DP(y,A,Th):
    r = y.clone()
    
    lossAll = []
    n = np.shape(A)[1]
    Sup = paddle.zeros((W,n),dtype='bool')
    WAll = paddle.zeros([W,n])
    for ii in range(200):
       print(ii)
       Z = paddle.abs(paddle.matmul(r, A))
       Z = Z/paddle.max(Z,axis=1,keepdim=True)

       pos = (Z>0.7)

       Sup = paddle.bitwise_or(pos, Sup)
       
       for jj in range(W):
           weight = paddle.matmul(paddle.linalg.pinv(A[:,Sup[jj,:]]), y[jj,:])
           r[jj,:] = y[jj,:] - paddle.matmul(A[:,Sup[jj,:]],weight)
           
           WAll[jj,Sup[jj,:]]=weight

       Loss = paddle.mean(r**2)

       lossAll.append(Loss.numpy())
       if Loss<Th:
           
           break
    
    return WAll, lossAll

Ts = time.time()


WeiAll, lossOMPp = OMP1DP(yT,AT,Th=StopTh)

sRecP = WeiAll.numpy()
sRecP = sRecP.transpose()

TimeAll[1]=time.time()-Ts
print(time.time()-Ts)


xOMP = np.matmul(D, sRecP)
print(np.mean(ssim1D(RFun,xOMP)))
#%% IHT in paddle GPU
def IHTp(y,A,Th):
    r = y.clone()

    u = 0.1
    lossIHT = []
  
    W, m = np.shape(y)
    K = np.int32(m/5)
    
    s = paddle.zeros([W,n],dtype='float32')
   
    for ii in range(200):
       print(ii)
       tmp = paddle.matmul(r,A)

       s = s + u*tmp

       AbsTmp = paddle.abs(s)
       Idx = paddle.argsort(AbsTmp,axis=-1)
       Idxno = Idx[:,0:n-K]
       
       for jj in range(W):
           s[jj,Idxno[jj,:]]=0.0
           

       r = y - paddle.matmul(s,A,transpose_y=True)

       Loss = paddle.mean(r**2)
      
       lossIHT.append(Loss.numpy())
       
       if Loss<Th:
           break
    
    return s, lossIHT

Ts = time.time()


sIHTp, lossIHT = IHTp(yT,AT,Th=StopTh)
sIHT = sIHTp.numpy().transpose()
    
    

TimeAll[2]=time.time()-Ts
print('TimeIHT',time.time()-Ts)

xIHT = np.matmul(D, sIHT)

print(np.mean(ssim1D(RFun,xIHT)))

#%%

x_pred = np.matmul(D, np.transpose(sPred.numpy()))

imgPred = np.squeeze(x_pred)
imgOMP = np.squeeze(xOMP)
imgIHT = np.squeeze(xIHT)

img = np.squeeze(RFun)
imgOMP[imgOMP<-1]=-1
imgOMP[imgOMP>1]=1

imgPred[imgPred<-1]=-1
imgPred[imgPred>1]=1

imgIHT[imgIHT<-1]=-1
imgIHT[imgIHT>1]=1


a1 = np.zeros([1,9])
a1[0,0:3]=TimeAll
a1[0,3]=np.mean(ssim1D(img,imgPred))
a1[0,4]=np.mean(ssim1D(img,imgOMP))
a1[0,5]=np.mean(ssim1D(img,imgIHT))
a1[0,6]=np.mean((img-imgPred)**2.0)
a1[0,7]=np.mean((img-imgOMP)**2.0)
a1[0,8]=np.mean((img-imgIHT)**2.0)

print('--Time:[CLOMP, OMP, IHT]----')
print(TimeAll)

print('--SSIM----')
print(a1[0,3:6])


print('--MSE----')
print(a1[0,6:9])

