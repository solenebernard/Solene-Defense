import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib
import matplotlib.patches as patches


import matplotlib.mlab as mlab
from scipy import stats
from PIL import Image
import rawpy
import imageio

import skimage.io
from scipy.fftpack import dct, idct
import os
import scipy.linalg as la
import scipy


rc('text', usetex=True)
rc('font', family='serif')

import numpy as np

s_size = 3



params_lin = rawpy.Params(rawpy.DemosaicAlgorithm.LINEAR, half_size=False, four_color_rgb=False, use_camera_wb=False, use_auto_wb=False, user_wb=(1,1,1,1), output_color=rawpy.ColorSpace.raw, output_bps=16, user_flip=None, user_black=0, user_sat=None, no_auto_bright=True, auto_bright_thr=None, adjust_maximum_thr=0.0, bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False, gamma=(1,1), chromatic_aberration=None, bad_pixels_path=None)
params_aahd = rawpy.Params(rawpy.DemosaicAlgorithm.AAHD, half_size=False, four_color_rgb=False, use_camera_wb=False, use_auto_wb=False, user_wb=(1,1,1,1), output_color=rawpy.ColorSpace.raw, output_bps=16, user_flip=None, user_black=0, user_sat=None, no_auto_bright=True, auto_bright_thr=None, adjust_maximum_thr=0.0, bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False, gamma=(1,1), chromatic_aberration=None, bad_pixels_path=None)
params_ppgnogamma = rawpy.Params(rawpy.DemosaicAlgorithm.PPG, half_size=False, four_color_rgb=False, use_camera_wb=False, use_auto_wb=False, user_wb=(1,1,1,1), output_color=rawpy.ColorSpace.raw, output_bps=16, user_flip=None, user_black=0, user_sat=None, no_auto_bright=True, auto_bright_thr=None, adjust_maximum_thr=0.0, bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False, gamma=(1,1), chromatic_aberration=None, bad_pixels_path=None)
params_ppggamma = rawpy.Params(rawpy.DemosaicAlgorithm.PPG, half_size=False, four_color_rgb=False, use_camera_wb=False, use_auto_wb=False, user_wb=(1,1,1,1), output_color=rawpy.ColorSpace.raw, output_bps=16, user_flip=None, user_black=0, user_sat=None, no_auto_bright=True, auto_bright_thr=None, adjust_maximum_thr=0.0, bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False, chromatic_aberration=None, bad_pixels_path=None)
params_vng = rawpy.Params(rawpy.DemosaicAlgorithm.VNG, half_size=False, four_color_rgb=False, use_camera_wb=False, use_auto_wb=False, user_wb=(1,1,1,1), output_color=rawpy.ColorSpace.raw, output_bps=16, user_flip=None, user_black=0, user_sat=None, no_auto_bright=True, auto_bright_thr=None, adjust_maximum_thr=0.0, bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False, gamma=(1,1), chromatic_aberration=None, bad_pixels_path=None)





# Quant table at 100% (convert)
c_quant_100 = np.array([\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1],\
        [ 1,  1,  1,  1,  1,  1,  1,  1]])


# Quant table at 95% (convert)
c_quant_95 = np.array([\
        [ 2,  1,  1,  2,  2,  4,  5,  6],\
        [ 1,  1,  1,  2,  3,  6,  6,  6],\
        [ 1,  1,  2,  2,  4,  6,  7,  6],\
        [ 1,  2,  2,  3,  5,  9,  8,  6],\
        [ 2,  2,  4,  6,  7, 11, 10,  8],\
        [ 2,  4,  6,  6,  8, 10, 11,  9],\
        [ 5,  6,  8,  9, 10, 12, 12, 10],\
        [ 7,  9, 10, 10, 11, 10, 10, 10]])

# Quant table at 85% (convert)
c_quant_85 = np.array([\
     [ 5,  3,  3,  5,  7, 12, 15, 18],\
     [ 4,  4,  4,  6,  8, 17, 18, 17],\
     [ 4,  4,  5,  7, 12, 17, 21, 17],\
     [ 4,  5,  7,  9, 15, 26, 24, 19],\
     [ 5,  7, 11, 17, 20, 33, 31, 23],\
     [ 7, 11, 17, 19, 24, 31, 34, 28],\
     [15, 19, 23, 26, 31, 36, 36, 30],\
     [22, 28, 29, 29, 34, 30, 31, 30]])

# Quant table at 75% (convert)
c_quant_75 = np.array([\
        [ 8,  6,  5,  8, 12, 20, 26, 31],\
        [ 6,  6,  7, 10, 13, 29, 30, 28],\
        [ 7,  7,  8, 12, 20, 29, 35, 28],\
        [ 7,  9, 11, 15, 26, 44, 40, 31],\
        [ 9, 11, 19, 28, 34, 55, 52, 39],\
        [12, 18, 28, 32, 41, 52, 57, 46],\
        [25, 32, 39, 44, 52, 61, 60, 51],\
        [36, 46, 48, 49, 56, 50, 52, 50]])

def dct2(x):
    return dct(dct(x, norm='ortho').T, norm='ortho').T
    
def idct2(x):
    return idct(idct(x, norm='ortho').T, norm='ortho').T

# Convert a TIFF image coded in 16 bits to a greyscale matrix coding in 16 bits too
def tiff2grey_16(imagefile):
    img = skimage.io.imread(imagefile+'.tiff', plugin='tifffile')
    img_grey = (img[:,:,0]).astype('uint32') * 299/1000 + (img[:,:,1]).astype('uint32') * 587/1000 + (img[:,:,2]).astype('uint32') * 114/1000
    print(img_grey.shape)
    
    img_grey[img_grey>2**16-1]=2**16-1
    return(img_grey.astype(np.int32))

# Convert a TIFF image coded in 16 bits to a greyscale matrix coding in 16 bits too
def tiff2grey_16_scale(imagefile):
    img = skimage.io.imread(imagefile+'.tiff', plugin='tifffile')
    img_grey = (img[:,:,0]).astype('uint32') * 299/1000 + (img[:,:,1]).astype('uint32') * 587/1000 + (img[:,:,2]).astype('uint32') * 114/1000
    print(img_grey.shape)

    img_s = np.array(Image.fromarray(img_grey).resize(size=(2*(1024+512),2*1024), resample=Image.LANCZOS))
    #img_s = np.array(Image.fromarray(img_grey).resize(size=(2*(1024+512),2*1024), resample=Image.BICUBIC))


    
    img_s[img_s>2**16-1]=2**16-1
    return(img_s.astype(np.int32))



# Compute DCT-Quantized coefficients from NON-quantized ones 
def compute_jpeg_from_dct(dct_im,c_quant):
    """
    Compute the jpeg representation from the DCT coefficients 
    """
    w,h = dct_im.shape
    jpeg_im = np.zeros((w,h))
    for bind_i in range(w//8):
        for bind_j in range(h//8):
            dct_bloc = dct_im[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8]
            jpeg_im[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8] += \
            np.round(dct_bloc/(c_quant*256))
    jpeg_im = jpeg_im.astype(np.int32)       
    return jpeg_im
   
    
def raw_to_dct(raw,imagefile,params):
    rgb = raw.postprocess(params)
    skimage.io.imsave(imagefile+'.tiff', rgb)
    im_grey = tiff2grey_16(imagefile)
    #print im_grey[:10,:10]
    dct_im = compute_dct_domain(im_grey)
    os.remove(imagefile+'.tiff')
    return(dct_im)

def raw_to_dct_scale(raw,imagefile,params):
    rgb = raw.postprocess(params)
    skimage.io.imsave(imagefile+'.tiff', rgb)
    im_grey = tiff2grey_16_scale(imagefile)
    #print im_grey[:10,:10]
    dct_im = compute_dct_domain(im_grey)
    os.remove(imagefile+'.tiff')
    return(dct_im)



def compute_dct_domain(im_pix):
    """
    Convert the image into DCT coefficients without performing quantization
    """
    w,h = im_pix.shape
    dct_im = np.zeros((w,h))
    for bind_i in range(w//8):
        for bind_j in range(h//8):
            im_bloc = im_pix[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8]
            dct_im[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8] = dct2(im_bloc-2**15-64)
    return dct_im


im_name = './DSC03334.ARW'
raw = rawpy.imread(im_name)

im_cover = raw.raw_image

print(np.max(im_cover))
std = 0.0005
im_cover[:,::2] = 2**8-5 
im_cover[:,1::2] = 2**8-5 

im_cover = im_cover + im_cover*std*np.random.randn(4024,6048)

raw.raw_image[:,:] = im_cover[:,:]

im_dct_lin = raw_to_dct_scale(raw,im_name,params_lin)
print(im_dct_lin.shape)

(h,w) = im_dct_lin.shape

(N_r,N_c)=(h,w)
print((N_r,N_c))

n_bloc = 2

obs_DCT_cover = np.zeros(((N_r//(8*n_bloc))*(N_c//(8*n_bloc)),64*n_bloc**2))
im_dct_lin[::8,::8] = 0
for i_bloc in range(N_r//(8*n_bloc)):
    for j_bloc in range(N_c//(8*n_bloc)):#scan left to right
        for i in range(n_bloc):
            for j in range(n_bloc):#scan left to right, starting from the top left
                obs_DCT_cover[(i_bloc)*(N_c//(8*n_bloc))+j_bloc,(i*n_bloc+j)*64:(i*n_bloc+j+1)*64] = \
                (im_dct_lin[(i_bloc*n_bloc+i)*8:(i_bloc*n_bloc+i+1)*8,(j_bloc*n_bloc+j)*8:(j_bloc*n_bloc+j+1)*8]).flatten() 

print(obs_DCT_cover.shape)


def plot_cov(cov,index_fig):
    fig = plt.figure(index_fig,figsize=(9*64.0/100,9*64.0/100), dpi=1000,frameon=False)
    max_val = np.max(cov)
    plt.imshow(cov,cmap='RdBu_r',vmin=-max_val, vmax=max_val)
    plt.colorbar()
    return fig

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

cov = np.cov(obs_DCT_cover.T[:,:])
mean = np.mean(obs_DCT_cover,axis=0)
#fig = plt.figure(0,figsize=(9*64.0/100,9*64.0/100), dpi=1000,frameon=False)
max_val = np.max(cov)/10
#plt.imshow(cov[:64],cmap='RdBu_r',vmin=-max_val, vmax=max_val)

fig, ax = plt.subplots(1,1,figsize=(20,4))
im = ax.imshow(cov[:64], cmap='RdBu_r',vmax=max_val, vmin=-max_val, interpolation=None)

ax.set_xticks(np.arange(5)*64-0.5)
ax.set_yticks(np.arange(2)*64-0.5)
ax.set_xticklabels(np.arange(5)*64)
ax.set_yticklabels(np.arange(2)*64)

for i in range(4):
    for j in range(1):
        rect = patches.Rectangle(np.array([i*64,j*64])-0.5, 64, 64, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
fig.colorbar(im,drawedges=False)
fig.savefig('../images/cov_BOSS.pdf', bbox_inches='tight', format='pdf')

plt.show()