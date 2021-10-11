import sys
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
sys.path.append('../../Manuscript/phd-thesis/figures/')
from tools import compute_jpeg_domain, compute_spatial_from_jpeg, compute_proba, c_quant_50
from scipy.stats import norm
from matplotlib import rc
#from tools_uniward import *


def quantization_matrix(f):
    if(f>50):
        s = 200-2*f
    else:
        s = 5000/f
    if(f==100):
        return(np.ones((8,8)))
    return(np.asarray(np.floor((50+s*c_quant_50)/100),dtype=np.int16))


def draw_modification(p):
    u = np.random.uniform(0, 1, p.shape)
    print((-np.log(-np.log(u)) + np.log(p+1e-30)).shape)
    b = np.argmax(-np.log(-np.log(u)) + np.log(p+1e-30), axis=0) - 1
    b = b.astype(np.float32)
    return(b)

H1 = 4 * np.array([[-0.25, 0.5, -0.25], 
            [0.5, -1, 0.5], 
            [-0.25, 0.5, -0.25]])
L1 = (1.0/9.0)*np.ones((3, 3))
L2 = (1.0/225.0)*np.ones((15, 15))

def hill_cost(X_spat):
    H1 = 4 * np.array([[-0.25, 0.5, -0.25], 
            [0.5, -1, 0.5], 
            [-0.25, 0.5, -0.25]])
    L1 = (1.0/9.0)*np.ones((3, 3))
    L2 = (1.0/225.0)*np.ones((15, 15))
    R = convolve2d(X_spat, H1.reshape((3,3)), mode = 'same', boundary = 'symm')
    # Low pass filter L1
    xi = convolve2d(abs(R), L1.reshape((3,3)), mode = 'same', boundary = 'symm')
    inv_xi = 1/(xi+1e-20)
    # Low pass filter L2
    rho = convolve2d(inv_xi, L2.reshape((15,15)), mode = 'same', boundary = 'symm')
    return(rho)


X_spat = np.array(Image.open('../../Manuscript/photos/BBQSigma2019/L1064275.jpg').convert('L'))
# H,W = X_spat.shape
# im_shape = min(H,W)
# diff = W - im_shape
# X_spat = (X_spat[:,diff//2:-diff//2]).astype('float32')
# X_spat = Image.fromarray(X_spat)
# X_spat = np.round(np.array(X_spat.resize((512,512))))
# c_quant = quantization_matrix(60)
# X_jpeg = compute_jpeg_domain(X_spat,c_quant)
# X_spat = compute_spatial_from_jpeg(X_jpeg,c_quant)



m = np.random.randint(0,2,int(1*X_spat.size))
rand = np.random.permutation(X_spat.size)
y = np.copy(X_spat.flatten()[rand]).astype(np.int16)
y[:len(m)] = y[:len(m)] - y[:len(m)]%2 + m
y = y[np.argsort(rand)].reshape(X_spat.shape)

fig, axs = plt.subplots(1,2, figsize=(6,2))
axs[0].hist(X_spat.flatten(),bins=10 ,range=(9.5,19.5),width=0.9)
axs[1].hist(y.flatten(),bins=10,range=(9.5,19.5),align='mid',width=0.9)

axs[0].set_title(r'Cover')
axs[1].set_title(r'Stego')
axs[0].set_ylabel(r'Number of pixels')

for ax in axs: 
    ax.set_xticks(np.arange(10,20,2))
    ax.set_xlabel('Pixel value')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_yticks([])
plt.savefig('../images/histograms.pdf', format='pdf')
plt.show()