def calculate_sparsity_rate_1d(data_matrix, threshold=0.98):
#calculate sparse rate on DCT
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