# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:23:49 2024

@author: BillT
"""


import paddle
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

import time
import cv2
# paddle.set_device('gpu:2')

img1 = cv2.imread("~/Data/1_1920x1080.tif",cv2.IMREAD_UNCHANGED)
print(np.shape(img1))
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)  # to gray

input_img = img1
H, W = np.shape(input_img)

# ---normalise
img = input_img.astype('float32')
img = img/np.max(img)  # [0,1]


n = H  #number of sampling points
r = 0.3
m = np.int32(n*r)


#%% basic functions

def GenerateMeaMtx(dataN, SampPer, Type='Bernoulli'):
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
        CNorm = 1.0/np.linalg.norm(MeaMtx,axis=0)
        MeaMtx = MeaMtx*CNorm
        

    return MeaMtx


# %% some global constant
import scipy.fftpack as fft


Mtx = GenerateMeaMtx(n,r,'Gaus')

Y = np.matmul(Mtx, img)


d1 = np.diag(np.ones([n]))
DCTBase = fft.dct(d1, axis=0, norm='ortho').transpose()



D = DCTBase
A=np.matmul(Mtx, D)
ANormSeed = 1.0/np.linalg.norm(A,axis=0)

A = A*ANormSeed

AT = paddle.to_tensor(A[:,:n]).astype('float32')
yT = paddle.to_tensor(Y).astype('float32').transpose(perm=[1,0])
DT = paddle.to_tensor(D).astype('float32').transpose(perm=[1,0])
TimeAll = np.zeros([3])
#%%
import paddle.nn as nn
import paddle.nn.functional as F
# paddle.set_device('gpu:2')

class CSRec1(nn.Layer):
    def __init__(self, A, y, neurons=32):
        super(CSRec1, self).__init__()
        self.m, self.n = np.shape(A)
        
        self.A = A
        self.y = y
        self.neurons = neurons
        self.fc1 = nn.Linear(in_features=self.m, out_features=self.neurons)#out_features=self.neurons)
        # self.fc2 = nn.Linear(in_features=self.neurons, out_features=self.neurons)#optional
        # self.fc3 = nn.Linear(in_features=self.neurons, out_features=self.neurons)#optional
        self.fc4 = nn.Linear(in_features=self.neurons, out_features=self.n)

        
    def forward(self,res, WSup, Th=0.7):
        h = F.relu(self.fc1(self.y))#s, sparse coefficients for just 1 neural layer
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)#s, sparse coefficients
        
        corr = paddle.abs(paddle.matmul(res, self.A))#method 1
        
        corr = corr/paddle.max(corr,axis=1,keepdim=True)
        Mask = (corr>Th)
        Mask = paddle.bitwise_or(Mask, WSup.astype('bool'))
       
        h0 = h*Mask.astype('float32')
        
        y_pred = paddle.matmul(h0,self.A,transpose_y=True)#method 1
        
        return Mask, y_pred, h
    
modelTest = CSRec1(AT, yT, neurons=128)
paddle.summary(modelTest,input_size=[(W,m),(W,n)],dtypes='float32')
#================================================================================
## This is the best one by our test
class CSRec2(nn.Layer):
    def __init__(self, A, W):
        super(CSRec2, self).__init__()
        self.m, self.n = np.shape(A)

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

#================================================================================
class CSRec3(nn.Layer):
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
        self.param1 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(self.A),trainable=False)
        self.fcRes = nn.Linear(in_features=self.m, out_features=self.n, weight_attr=self.param1, bias_attr=False)
        
        self.param = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.transpose(self.A,perm=[1, 0])),trainable=False)
        self.fcLast = nn.Linear(in_features=self.n, out_features=self.m, weight_attr=self.param, bias_attr=False)

        
    def forward(self,res, WSup, Th=0.7):
        h = F.relu(self.fc1(self.y))#s, sparse coefficients for just 1 neural layer
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)#s, sparse coefficients
        
        corr = paddle.abs(self.fcRes(res))
        
        corr = corr/paddle.max(corr,axis=1,keepdim=True)
        Mask = (corr>Th)
        Mask = paddle.bitwise_or(Mask, WSup.astype('bool'))
       
        h0 = h*Mask.astype('float32')
        
        y_pred = self.fcLast(h0)
        
        
        return Mask, y_pred, h
    
modelTest = CSRec3(AT, yT, neurons=128)
paddle.summary(modelTest,input_size=[(W,m),(W,n)],dtypes='float32')
#%%
def train(net, optimizer, yIn, epoch_num = 10):
    lossAll = []

    NewS = []
    # global LRs

    print('Start training...')
    residual = yIn
    net.train()

    WSup = paddle.zeros((1,n),dtype='bool')
    #
    ThNew=0.7

    for epoch in range(epoch_num):
        WSup, y_pred, s = net(residual,WSup,ThNew)
        
        residual = yIn - y_pred
        NewS.append(paddle.sum(WSup).numpy())
        
        loss1 = F.mse_loss(y_pred, yIn,reduction='mean')

        imgPred = paddle.matmul(s,DT)
        
        
        Var =  (paddle.pow(imgPred[1:,:] - imgPred[:(W-1),:], 2).mean() )+\
                   (paddle.pow(imgPred[:,1:] - imgPred[:,:(H-1)], 2).mean())

        ImgBlock= F.unfold(paddle.unsqueeze(imgPred, axis=[0,1]),[3,3],strides=2,paddings=0)
        
        loss2 = paddle.mean(paddle.std(ImgBlock,axis=[1]))

        
        loss = loss1 +0.1*Var + 0.2*loss2 #

        #---------------step
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        lossNP = loss.numpy()
        lossAll.append(lossNP)
        Lr.step()
        
        print('epoch = %d    loss = %.6f   lr = %.5f' % (epoch, lossNP, optimizer.get_lr()))
        #####----------------------------------
        if epoch>80:
            if (loss1.numpy()<1e-4)|(epoch==epoch_num-1):#|(np.abs(lossAll[-2]-lossAll[-1])<1e-7):#quick visualisation
                break
        
    return WSup, lossAll,s,NewS

#---------------------------------

Ts = time.time()

# model = CSRec3(AT,yT,neurons=128)#try 0.1*Var + 0.2*loss2, learning rate =0.01 to 0.001
# model = CSRec3(AT,yT,neurons=128)#try 0.1*Var + 0.2*loss2, learning rate =0.01 to 0.001
model = CSRec2(A=AT,W=W)#try 0.1*Var + 0.2*loss2, learning rate =2 to 0.001
Lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=2, decay_steps=150,end_lr=0.001)

# nn.initializer.set_global_initializer(nn.initializer.Constant(0.2), nn.initializer.Constant(0.2))#option: use this for CSRec2

Opt = paddle.optimizer.Adam(learning_rate=Lr, parameters=model.parameters())

Enum = 150
Sup,lossAll,sPred,NewS =train(net=model, optimizer=Opt, yIn=yT,epoch_num=Enum)

TimeAll[0]=time.time()-Ts
print('Time: ',TimeAll[0])

# print(np.sum(Sup.numpy()))


plt.figure()
plt.semilogy(lossAll,'--r')

plt.title('loss curve')
plt.grid(visible=True)
plt.show()

x_pred = np.matmul(D, np.transpose(sPred.numpy()))
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
       
       
       Z = Z/(paddle.max(Z,axis=1,keepdim=True)+1e-6)
       
       pos = (Z>0.7)
       

       Sup = paddle.bitwise_or(pos, Sup)

       for jj in range(W):
 
           ZeroDet = paddle.sum(Sup[jj,:])
           
           if ZeroDet!=0:
               
               pos1 = Sup[jj,:]
               weight = paddle.matmul(paddle.linalg.pinv(A[:,pos1]), y[jj,:])
               r[jj,:] = y[jj,:] - paddle.matmul(A[:,pos1],weight)
               
               WAll[jj,pos1]=weight

       Loss = paddle.mean(r**2)

       lossAll.append(Loss.numpy())
       
       if Loss<Th:
           
           break
    
    return WAll, lossAll

Ts = time.time()


WeiAll, lossOMPp = OMP1DP(yT,AT,Th=1e-4)

sRecP = WeiAll.numpy()
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


sIHTp, lossIHT = IHTp(yT,AT,Th=1e-4)
sIHT = sIHTp.numpy().transpose()
    
    

TimeAll[2]=time.time()-Ts
print(TimeAll[2])
# print(np.shape(SupPos))

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

x_pred = np.matmul(D, np.transpose(sPred.numpy()))


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
