import sys
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
sys.path.append('../../Manuscript/phd-thesis/figures/')
from tools import compute_jpeg_domain, compute_spatial_from_jpeg, compute_proba, c_quant_50, draw_modification
from scipy.stats import norm
from matplotlib import rc
#from tools_uniward import *

# tuple(np.array([126,179,86])/255) # light green
# tuple(np.array([19,28,13])/255) # dark green
c = [tuple(np.array([126,179,86])/255), (1.,1.,1.), tuple(np.array([234,144,16])/255)]
cmap = LinearSegmentedColormap.from_list('greenorange', c, N=100)


def draw_modification(p, im_size):
    u = np.random.uniform(0, 1, (3, im_size, im_size))
    b = np.argmax(-np.log(-np.log(u)) + np.log(p+1e-30), axis=0) - 1
    b = b.astype(np.float32)
    return(b)


def quantization_matrix(f):
    if(f>50):
        s = 200-2*f
    else:
        s = 5000/f
    if(f==100):
        return(np.ones((8,8)))
    return(np.asarray(np.floor((50+s*c_quant_50)/100),dtype=np.int16))


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
H,W = X_spat.shape
im_shape = min(H,W)
diff = W - im_shape
X_spat = (X_spat[:,diff//2:-diff//2]).astype('float32')
X_spat = Image.fromarray(X_spat)
X_spat = np.round(np.array(X_spat.resize((512,512))))
c_quant = quantization_matrix(60)
X_jpeg = compute_jpeg_domain(X_spat,c_quant)
X_spat = compute_spatial_from_jpeg(X_jpeg,c_quant)

# nz_AC = np.sum(X_jpeg==0)-np.sum(X_jpeg[::8,::8]==0)
# juni = J_UNIWARD_process('',0.4).compute_rhos(X_spat, X_jpeg, c_quant)
# juni = np.concatenate((juni[1][None],np.zeros((1,)+juni[0].shape),juni[0][None]))


fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(X_spat,cmap='gray')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('../images/estonie.pdf', format='pdf')
plt.show()

rho = hill_cost(X_spat)

fig, axs = plt.subplots(1,3,figsize=(8,3))
axs[0].imshow(X_spat,cmap='gray')
axs[0].set_title('Cover')
axs[1].imshow(X_spat%2,cmap='gray')
axs[1].set_title('LSB of Cover')
axs[2].imshow(np.zeros((512,512)),alpha=0)
for ax in axs:
    ax.set_axis_off()
plt.tight_layout()
plt.savefig('../images/estonie_LSB.pdf', format='pdf')
plt.show()


fig, axs = plt.subplots(1,3,figsize=(8,3))
axs[0].imshow(X_spat,cmap='gray')
axs[0].set_title('Cover')
axs[1].imshow(X_spat%2,cmap='gray')
axs[1].set_title('LSB of Cover')
axs[2].set_title('LSB of Stego')
axs[2].imshow(np.random.randint(0,2,size=(512,512)),cmap='gray')
for ax in axs:
    ax.set_axis_off()
plt.tight_layout()
plt.savefig('../images/estonie_random.pdf', format='pdf')
plt.show()




plt.figure(figsize=(8,3.6))

plt.subplot(121)
plt.axis('off')
plt.title('Cover')
plt.imshow(X_spat,cmap='gray')

plt.subplot(122)
plt.imshow(np.log(rho+1),vmax=0.5, cmap=cmap)
plt.axis('off')
plt.title(r'Cost map $\rho_i = \rho_i^{-1} = \rho_i^{+1}$')
plt.colorbar()

plt.tight_layout()
plt.savefig('../images/estoniecostmap.pdf', format='pdf')
plt.show()

rho3 = np.concatenate((rho[None],np.zeros((1,)+rho.shape),rho[None]))
probas = compute_proba(rho3, 0.1*X_spat.size)[0]
b = draw_modification(probas, probas.shape[1])

plt.figure(figsize=(12,2.2))

plt.subplot(151)
plt.axis('off')
plt.title(r'$\mathbf{x}$')
plt.imshow(X_spat,cmap='gray')

plt.subplot(152)
plt.imshow(np.log(rho+1),vmax=0.5, cmap=cmap)
plt.axis('off')
plt.title(r'$\{\rho_i\}$')
plt.colorbar()

plt.subplot(153)
plt.imshow(probas,cmap=cmap)
plt.axis('off')
plt.title(r'$\{\pi_i\}$')
plt.colorbar()

plt.subplot(154)
plt.imshow(b,cmap='gray')
plt.axis('off')
plt.title(r'$\{b_i\}$')
plt.colorbar()


plt.subplot(155)
plt.imshow(b+X_spat,cmap='gray')
plt.axis('off')
plt.title(r'$\mathbf{y}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('../images/estonie_pipeline.pdf', format='pdf')
plt.show()



rc('text', usetex=True)
rc('font',**{'family':'serif'})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
rc('font', **font)

X_spat = np.array(Image.open('../../Manuscript/photos/PragueNov19/piscine.jpg').convert('L'))
H,W = X_spat.shape
im_shape = min(H,W)
diff = W - im_shape
X_spat = X_spat[:,diff:]


plt.figure(figsize=(10,10))
plt.imshow(X_spat,cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('../images/piscine.pdf', format='pdf')
plt.show()

plt.figure(figsize=(5,5))
mat = X_spat[-8:,-8:]
plt.imshow(mat,cmap='gray',vmax=20,vmin=0)
for (j,i),label in np.ndenumerate(mat):
    plt.text(i,j,label,ha='center',va='center')
    plt.text(i,j,label,ha='center',va='center')
plt.axis('off')
plt.tight_layout()
plt.savefig('../images/zoom_spatial.pdf', format='pdf')
plt.show()


X_spat = np.round(np.array(Image.fromarray(X_spat).resize((512,512))))
X_jpeg = compute_jpeg_domain(X_spat,c_quant) 

plt.figure(figsize=(10,10))
plt.imshow(X_jpeg,cmap='gray',vmax=10,vmin=-10)
plt.axis('off')
plt.tight_layout()
plt.savefig('../images/piscine_jpeg.pdf', format='pdf')
plt.show()


plt.figure(figsize=(5,5))
mat = X_jpeg[-8*4:-8*3,-8*4:-8*3]
plt.imshow(mat,cmap='gray',vmax=90,vmin=-90)
for (j,i),label in np.ndenumerate(mat):
    plt.text(i,j,label,ha='center',va='center',c='black')
    plt.text(i,j,label,ha='center',va='center',c='black')
plt.axis('off')
plt.tight_layout()
plt.savefig('../images/zoom_jpeg.pdf', format='pdf')
plt.show()
