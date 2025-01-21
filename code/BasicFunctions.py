def calculate_sparsity_rate_1d(data_matrix, threshold=0.98):
#calculate sparse rate on DCT for 1D signals
    n, W = data_matrix.shape  # 
    sparsity_rates = []  # 

    for w in range(W):  # 
        column = data_matrix[:, w]  # 
        column = column.astype('float32')

        column = column / np.max(np.abs(column))

        S = np.abs(np.fft.fft(column, norm='ortho'))

        S_sorted = np.sort(S)[::-1]

        Csum = np.cumsum(S_sorted)
        Csum = Csum / np.sum(S_sorted)

        Pos = np.where(Csum > threshold)[0][0]

        sparsity_rate = Pos / n
        sparsity_rates.append(sparsity_rate)

    average_sparsity = np.mean(sparsity_rates)
    
    return average_sparsity


def calculate_sparsity_rate_2d(img1, threshold=0.98):
    """
    Calculate sparse rate of image on 2D FFT
    
    Par:
        img1: numpy image
        threshold: threshold value that indicates the main power
    
    Return：
        Sparse rate
    """
    if len(img1.shape) == 3: 
        input_img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        input_img = img1
    
    img = input_img.astype('float32')

    H, W = np.shape(img)
    img = img/np.max(img)#

    S = np.abs(np.fft.fft2(img,norm='ortho'))
    S1 = np.reshape(S,[1,-1])
    S1sorted = np.fliplr(np.sort(S1))

    Csum = np.cumsum(S1sorted)# (1, 8294400)
    Csum = Csum/np.sum(S1sorted)#

    Pos = np.where(Csum>threshold)[0][0]

    Kr = Pos/H/W
    
    return Kr

def ssim2D(img1, img2,DR):
  L = DR# depth of image (255 in case the image has a differnt scale)
  winS = 3
  C1 = (0.02 * L)**2
  C2 = (0.02 * L)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(winS, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)#[1:-1, 1:-1] # valid
  mu2 = cv2.filter2D(img2, -1, window)#[1:-1, 1:-1]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq#[5:-5, 5:-5]
  sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
  # ssim_map =  (2.0*sigma12 + C1)/(sigma1_sq + sigma2_sq + C2)
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

#calculate PSNR
psnr = 20.0 * np.log10(1.0 / np.sqrt(mse))

#calculate Pearson Correlation Coefficient
from scipy.stats import pearsonr
PCC, _ = pearsonr(GroundTruth, Predicted)
