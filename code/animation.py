import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib import rc
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

N = 5
np.random.seed(4)
x0 = np.random.randn(N,2)+3
x1 = np.random.randn(N,2)

x0 = np.array([[0, 3], [1,2], [2.5,2], [1.5,3], [2,1]])+np.random.randn(N,2)*0.1
x1 = np.array([[-1, 1.5], [-0.5,-0], [1,0], [-0.5, 1], [-1.5, 0]])+np.random.randn(N,2)*0.1

y0 = np.zeros(N)
y1 = np.ones(N)


b0 = np.array([[-2, -3], [-1,-2], [0,1], [1,2], [2,2], [4, 5]])
b1 = np.array([[-2,5],[-1,3],[0,2],[1,1], [2,0], [4, 1]])
xx = np.linspace(-2,4,101)

paris = [mpimg.imread('../animation_training/paris'+str(i)+'.jpg') for i in range(1,6)]
lille = [mpimg.imread('../animation_training/lille'+str(i)+'.jpg') for i in range(1,6)]
prague = [mpimg.imread('../animation_training/prague'+str(i)+'.jpg') for i in range(1,6)]


poly = lagrange(b0[:,0], b0[:,1])
yy0 = Polynomial(poly.coef[::-1])(xx)

poly = lagrange(b1[:,0], b1[:,1])
yy1 = Polynomial(poly.coef[::-1])(xx)

rc('text', usetex=True)
rc('font',**{'family':'serif'})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 60}
rc('font', **font)
green = tuple(np.array([126,179,86])/255)

for i,t in enumerate(np.linspace(0,1,31)[::-1]):
    fig, ax = plt.subplots(1,1,figsize=(30,20))
    plt.plot(xx,t*yy0+(1-t)*yy1,c=green,linewidth=5)
    for im,c in zip(lille, x0):
        imagebox = OffsetImage(im, zoom=0.4)
        ab = AnnotationBbox(imagebox, c, frameon = False)  
        ax.add_artist(ab)
    for im,c in zip(prague, x1):
        imagebox = OffsetImage(im, zoom=0.4)
        ab = AnnotationBbox(imagebox, c, frameon = False)  
        ax.add_artist(ab)
    plt.axis('off')
    plt.xlim(-2,3.5)
    plt.ylim(-1,4)
    plt.tight_layout()
    plt.title(r'Training of $f_\theta$')
    if(t==0):
        plt.title(r'Trained $f_{\theta^\star}$')
        plt.annotate("Prague", (2,-0.75))
        plt.annotate("Lille", (2.5,0))
    plt.savefig('../animation_training/'+str(i)+'.png', format='png')
    plt.show()


    
cover = [mpimg.imread('../animation_training_stega/'+str(i)+'.jpg') for i in range(1,6)]

for i,t in enumerate(np.linspace(0,1,31)[::-1]):
    fig, ax = plt.subplots(1,1,figsize=(30,20))
    plt.plot(xx,t*yy0+(1-t)*yy1,c=green,linewidth=5)
    for im, c0, c1 in zip(cover, x0, x1):
        imagebox = OffsetImage(im, zoom=0.4, cmap='gray')
        ab = AnnotationBbox(imagebox, c0, frameon = True)  
        ax.add_artist(ab)
        ab = AnnotationBbox(imagebox, c1, frameon = False)  
        ax.add_artist(ab)
    
    plt.axis('off')
    plt.xlim(-2,3.5)
    plt.ylim(-1,4)
    plt.tight_layout()
    plt.title(r'Training of $f_\theta$')
    if(t==0):
        plt.title(r'Trained $f_{\theta^\star}$')
        plt.annotate("Cover", (2,-0.75))
        plt.annotate("Stego", (2.5,0))
    plt.savefig('../animation_training_stega/'+str(i)+'.png', format='png')
    plt.show()
    
    

# Adversarial example
x0p = np.copy(x0)
x0p[0] = [-1.5,2.75]

for i,t in enumerate(np.linspace(0,1,31)[::-1]):
    fig, ax = plt.subplots(1,1,figsize=(30,20))
    plt.plot(xx,yy1,c=green,linewidth=5)
    for im, c0, c1, c0p in zip(cover, x0, x1, x0p):
        imagebox = OffsetImage(im, zoom=0.4, cmap='gray')
        ab = AnnotationBbox(imagebox, c0*t+(1-t)*c0p, frameon = True)  
        ax.add_artist(ab)
        ab = AnnotationBbox(imagebox, c1, frameon = False)  
        ax.add_artist(ab)
    
    plt.axis('off')
    plt.xlim(-2,3.5)
    plt.ylim(-1,4)
    plt.tight_layout()
    plt.title(r'Update of a cost map $\rho$')
    plt.annotate("Cover", (2,-0.75))
    plt.annotate("Stego", (2.5,0))
    plt.annotate(r'$f_{\theta^\star}$', (3,-0.6), c=green)
    plt.savefig('../animation_adversarial_stego/'+str(i)+'.png', format='png')
    plt.show()
    
    
    


# Retraining adversarial
b1p = np.array([[-2.5,2], [-2,2], [-1,2],[0,2],[1,1], [2,0], [4, 1]])
poly = lagrange(b1p[:,0], b1p[:,1])
yy1p = Polynomial(poly.coef[::-1])(xx)

for i,t in enumerate(np.linspace(0,1,31)[::-1]):
    fig, ax = plt.subplots(1,1,figsize=(30,20))
    plt.plot(xx,t*yy1+(1-t)*yy1p,c=green,linewidth=5)
    for im, c0, c1 in zip(cover, x0p, x1):
        imagebox = OffsetImage(im, zoom=0.4, cmap='gray')
        ab = AnnotationBbox(imagebox, c0, frameon = True)  
        ax.add_artist(ab)
        ab = AnnotationBbox(imagebox, c1, frameon = False)  
        ax.add_artist(ab)
    
    plt.axis('off')
    plt.xlim(-2,3.5)
    plt.ylim(-1,4)
    plt.tight_layout()
    plt.title(r'Training of $f_\theta$')
    if(t==0):
        plt.title(r'Trained $f_{\theta^\star}$')
        plt.annotate("Cover", (2,-0.4))
        plt.annotate("Stego", (2.6,1))
    plt.savefig('../animation_retraining_stega/'+str(i)+'.png', format='png')
    plt.show()
    
    
    

# Animation of the weights of the classifier


# Animation of gradient descent
def f(x):
    return(x**2)

def gradf(x):
    return(2*x)

x0 = -5
alpha = 5e-2
n = 30

x = x0
xx = np.linspace(-6, 4, 101)

for i in range(n):
    dx = gradf(x)
    x -= alpha * dx
    
    plt.plot(xx, f(xx))
    plt.scatter(x,f(x), s = 10, c= 'red')
    #plt.arrow(x, f(x), alpha, dx*alpha, width=0.1)
    plt.axis('off')
    plt.title('Optimization via gradient descent')
    plt.savefig('../animation_gradient_descent/'+str(i)+'.png', format='png')
    plt.show()



rc('text', usetex=True)
rc('font',**{'family':'serif'})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)

np.random.seed(42)
mat0 = np.random.randn(8,8)
mat1 = np.random.randn(8,8)

c = [tuple(np.array([126,179,86])/255), (1.,1.,1.), tuple(np.array([234,144,16])/255)]
cmap = LinearSegmentedColormap.from_list('greenorange', c, N=100)

n = 1
for k,t in enumerate(np.linspace(0,1,n)): 
    
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    
    mat = mat1 * np.sqrt(t) + mat0 * (1-np.sqrt(t))
    axs[1].imshow(mat,cmap=cmap,vmax=3,vmin=-3)
    for (j,i),label in np.ndenumerate(mat):
        axs[1].text(i,j,np.round(label,2),ha='center',va='center')
        rect = patches.Rectangle(np.array([i,j])-0.5, 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)
    #rect = patches.Rectangle(np.array([-0.5, 0.5]), 8, 8, linewidth=1, edgecolor='black', facecolor='none')
    #axs[0].add_patch(rect)
    
    axs[1].text(-1.35,3.5, r'$\star$')

    X_spat = np.array([[11, 11, 11, 12, 12, 13, 13, 13],
        [10, 10, 11, 11, 12, 12, 13, 13],
        [ 9, 10, 10, 10, 11, 11, 12, 12],
        [ 9,  9, 10, 10, 11, 11, 12, 12],
        [10, 10, 10, 11, 11, 12, 12, 12],
        [11, 11, 11, 12, 13, 13, 13, 14],
        [12, 13, 13, 13, 14, 15, 15, 15],
        [13, 14, 14, 14, 15, 15, 16, 16]], dtype=np.int8)

    axs[0].imshow(X_spat,cmap='gray',vmax=20,vmin=5)
    for (j,i),label in np.ndenumerate(X_spat):
        axs[0].text(i,j,np.round(label,2),ha='center',va='center')
        rect = patches.Rectangle(np.array([i,j])-0.5, 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        axs[0].add_patch(rect)
    #rect = patches.Rectangle(np.array([-0.5, 0.5]), 8, 8, linewidth=1, edgecolor='black', facecolor='none')
    #axs[1].add_patch(rect)

    for ax in axs:
        ax.axis('off')
    
    axs[0].set_title(r'Image')
    axs[1].set_title(r'Parameters $\theta$')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig('../animation_gradient_descent/'+str(k)+'.png', format='png')
    plt.show()
    
    




np.random.seed(43)
H = np.random.randint(0,2,(4,6))
x = np.random.randint(0,2,H.shape[1])
m = np.random.randint(0,2, H.shape[0])

r = []
for i in range(2**len(x)):
    bi = bin(i)[2:]
    bi = '0'*(len(x)-len(bi)) + bi
    y = np.array([int(b) for b in bi])
    if(np.sum(np.dot(H,y)%2!=m)==0):
        r.append(y)
        print((y-x)%2)