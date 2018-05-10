import warnings
warnings.filterwarnings('ignore')

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as mp
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')

from numpy.random import randn,rand
from numpy.linalg import norm, svd
from numpy.lib.stride_tricks import as_strided
from scipy.misc import imread,imsave
from time import clock

import utils as utl
import performance as pfm



def tasks_A_1(path):
    # TASK A-1.1
    print('\n'+'+++TASK A-1.1'*4)
    X = facesToMatrix(path+'/FACES-PNG/')
    xmean = np.average(X,axis=1)    
    Y = centerColumnData(X,xmean)
    U,d,Vt = svd(Y)
    print('Face image vectors are centered and',
          'KLT base is found by SVD')
    print('-'*40)   
    # TASK A-1.2
    print('\n'+'+++TASK A-1.2'*4)
    E = formEigenImage(xmean,U,7,7)
    imsave(path+'/eigenfaces.jpg',E)
    print('Eigenimage is created and saved to',
          path+'/eigenfaces.jpg')
    print('-'*40)
    # TASK A-1.3
    print('\n'+'+++TASK A-1.3'*4)
    U48 = U[:,:48]
    serializeToFile([xmean,U48],
            path+'/eigenfaces.pck')
    print('Model "PCA-48D and mean"',
         'is serialized and saved to',
         path+'/eigenfaces.pck')
    print('-'*40)
    # TASK A-1.4
    print('\n'+'+++TASK A-1.4'*4)
    xmeanp,Up = reflectEigenfaces(xmean,U48)    
    Ep = formEigenImage(xmeanp,Up,7,7)
    imsave(path+'/reflectedEigenfaces.jpg',Ep)
    serializeToFile([xmeanp,Up],
                path+'/reflected_eigenfaces.pck')
    print('For reflected facial images',
          'average eigenimage is created and saved',
          'into '+path+'/reflected_eigenfaces.pck')
    print('48 average eigenfaces are saved into',
          path+'/reflected_eigenfaces.pck')          
    print('-'*40)
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-1.5 (for one point)
    print('\n'+'+++TASK A-1.5'*4)
    cos_tab = degreeOfOrthogonality(Up)
    imsave(path+'/'+'degree-of-orthogonality.jpg', scaleMatrix(cos_tab) )
    mp.imshow(cos_tab); mp.show()
    print('Minimal |cos_tab|', np.min(np.abs(cos_tab)))
    print('='*60)
    print('Image of cosine values for angles',
     'between average eigenfaces is saved to',
     path+'/degree-of-orthogonality.jpg')
    
    print('='*60)
def facesToMatrix(faces_path, k=5, mu_yes = False,
                  start=1,step=4,stop=636,
                  xres=46,yres=56):
    N = xres*yres; 
    P = 1+(stop-1-start)//step; L = P*k
    X = np.zeros((N,L))    
    j = 0
    for p in range(start,stop,step):
        for i in range(1,6):
            name = 'mpeg_{0:04d}_{1:04d}.png'\
                   .format(p,i)
            mat = imread(faces_path+name)
            #print(mat.shape,mat.dtype)
            mat = np.mean(mat,axis=2)                     
            #mp.imshow(mat); mp.show()
            X[:,j] = mat.flatten();  j += 1
    if mu_yes:
        mu  = np.kron(range(P),
                 np.ones(k,dtype=np.int32))
        return X,mu
    else:        
        return X
def centerColumnData(X,xmean):
    Y = X-as_strided(xmean,shape=X.shape,
                     strides=(xmean.strides[0],0))
    return Y
def scaleVector(e):
    se = e-np.min(e)
    se /= np.max(se)
    se *= 255
    return se
def formEigenImage(xmean,U,M,N):
    global w,h
    w =  46; h = 56
    E = np.zeros((h*M,w*N),dtype=np.uint8)
    
    k = 0
    for m in range(M):
        for n in range(N):
            if k==U.shape[1]: break
            e = U[:,k]; k += 1
            se = scaleVector(e)
            E[m*h:(m+1)*h,n*w:(n+1)*w] =\
                                 se.reshape(h,w)
    E[-h:,-w:] = scaleVector(xmean).reshape(h,w)
    return E
def serializeToFile(obj,name,path='.'):
    from pickle import dump
    fp = open(name,'wb')
    dump(obj,fp,protocol=2)
    fp.close()
def reflectEigenfaces(xmean,U):
    N,n = U.shape
    Up = np.zeros(U.shape)    
    for i in range(n):
        f = U[:,i]
        F = f.reshape(h,w)
        Fp = np.fliplr(F)
        G = (F+Fp)/2.0
        Up[:,i] = G.reshape(h*w)
    xmeanp = np.fliplr(xmean.reshape(h,w)).reshape(h*w)
    gmean = (xmean+xmeanp)/2.
    return gmean,Up
def scaleMatrix(A):
    As = A-np.min(A)*np.ones(A.shape)
    As /= np.max(As)
    return np.array(As*255,dtype=np.uint8)
def degreeOfOrthogonality(Up):
    D = np.dot(Up.T,Up)
    dgl = np.sqrt(np.diag(D))
    D /= np.outer(dgl,dgl)
    return D
def tasks_A_2(path):
    # TASK A-2.1
    print('\n'+'+++TASK A-2.1'*4)
    Z = facesToMatrix(path+'/FACES-PNG/',start=3)
    xmean,U = deSerializeFromFile(path+\
                      '/eigenfaces.pck')
    Zc = centerColumnData(Z,xmean)
    F = np.dot(U.T,Zc)
    Zp = np.dot(U,F)+\
         np.outer(xmean,np.ones(Z.shape[1]))
    Zpca = formRandomPCAImage(Z,Zp,N=20)    
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-2.1 (for one point):
    ### YOUR CODE HERE
    #     code to save the reconstructed images 
    ###
    print('-'*40)
    print('The image of 20 original faces and'+\
          ' their reconstructions from PCA-48D'+\
          ' .created and saved to the file.')
    print('-'*40)
    # TASK A-2.2
    print('\n'+'+++TASK A-2.2'*4)
    mse_ortho = reconstructionMSE(Zc,U,kmax=30)
    gmean,Up = deSerializeFromFile(path+\
                    '/reflected_eigenfaces.pck')    
    Zcp = centerColumnData(Z,gmean)
    mse_non_ortho = reconstructionMSE(Zcp,Up,
                                      kmax=30)
    K_max = len(mse_ortho)
    Ks = range(1,K_max+1)
    fig = mp.figure()
    mp.plot(Ks,mse_ortho,'b',
               label='classical PCA')
    mp.plot(Ks,mse_non_ortho,'g',
               label='modified PCA')
    mp.legend(loc='upper right')
    title = 'Reconstruction MSE wrt PCA dimension'
    mp.title(title)
    fig.canvas.set_window_title(title)
    mp.draw(); mp.show()
    print('Classical PCA was compared with'+\
      'modified PCA wrt reconstruction MSE'+\
            'at PCA dimension K=1,...,{:d}'.\
            format(K_max))    
    print('='*60)
def deSerializeFromFile(name,path='.'):
    from pickle import load
    fp = open(path+'/'+name,'rb')
    obj = load(fp)
    fp.close()
    return obj
def formRandomPCAImage(Z,Zp,N=5):
    idx = rn.randint(0, Z.shape[1],N)
    A = Z[:,idx]; B = Zp[:,idx]    
    w =  46; h = 56
    Iab = np.zeros((h*2,w*N),dtype=np.uint8)
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-2.1 continuation:
    #      code to fill up the array Iab
def reconstructionMSE(Zc,U,kmax=None):
    N,Kmax = U.shape
    if kmax==None: kmax=Kmax
    L = Zc.shape[1]
    mse = []
    for K in range(1,kmax+1):
        U_K = U[:,:K]; F_K = np.dot(U_K.T,Zc)
        Zp = np.dot(U_K,F_K)        
        nm = norm(Zc-Zp)
        mse.append(nm*nm/L)
    return mse
def tasks_A_3(path):
    # TASK A-3.1
    print('\n'+'+++TASK A-3.1'*4)
    Z = facesToMatrix(path+'/FACES-PNG/',start=3)
    xmean,U = deSerializeFromFile(path+\
                            '/eigenfaces.pck')
    Zc = centerColumnData(Z,xmean)
    F = np.dot(Zc.T,U)
    L = Z.shape[1]; K = 5; P = L//K
    mu = utl.uniformMembership(P,K)
    alphas=[1,0.5]
    plotPerformanceMeasures(F,mu,alphas=alphas,
        title='Face recognition for Euclidean'+\
    ' and combined proximity with weight 0.5',
        win_title='TASK A-3.1')
    print('Performance measures were plotted'+\
      ' for Euclidean and combined proximity'+\
      ' with weight 0.5')
    print('-'*40)            
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-3.2: TO DO for one point
    print('\n'+'+++TASK A-3.2'*4)
    alphas=[0.0, 0.25, 0.5, 0.75, 1.0]
    ### YOR CODE HERE
    #   call plotPerformanceMeasures
    ###
    print('Performance measures were plotted for'+\
    ' combined measures with alpha in '+str(alphas))
    print('='*60)
    return Z,F,mu
def tasks_A_4(Z,F,mu,path):
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-4.1: TO DO for one point
    print('\n'+'+++TASK A-4.1'*4)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    ### YOUR CODE HERE:
    #   code for the modified PCA features (Fp)
    #   and call plotPerformanceMeasures
    ###
    print('Performance measures SR,RC,PR,'+\
    'F_1,F_2,ANMRR are computed and plotted'+\
    '\nfor combined proximities with '+\
    'alpha in '+str(alphas)+\
    '\nwhile features are average eigenfaces')
    print('-'*40)        
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK A-4.2: TO DO for three points
    print('\n'+'+++TASK A-4.2'*4)
    ### YOUR CODE HERE: 
    #   code to generate performance measures
    # (a) for weights in [0,0.5,1]
    # (b) for PCA and modified PCA with m
    #     components, m in [3,6,...,48]
    # Plot the generated measures in one figure
    # by different colors interpreted by legend 
    ###
    print('Success rate is plotted wrt '+\
          'classical versus modified PCA for'+\
          ' proximities with weights: 0,.5,1')
    print('='*60)
def plotPerformanceMeasures(F,mu,alphas=None,kNN=None,
             title=None,win_title=None,xvalues=None,
             lrow=-1,lcol=-1):
    pp = utl.ProximityPlus(F,mu)
    if type(xvalues)==type(None):
        if kNN: kmax = kNN
        else: kmax = F.shape[0]-1
        x_v = np.arange(1,kmax+1)
        x_vals = [[x_v]*3,[x_v]*3]
    else:
        kmax = len(xvalues)
        x_vals = [[xvalues]*3,[xvalues]*3]
    y_vals = []
    if type(alphas)==type(None):
        alphas = [1.0]
    labels = [] 
    titles = [
       ['Success rate k-NN','Recall k-NN',
        'Precision k-NN'],
       ['Measure F1', 'Measure F2',
        'Rank measure ANMRR']
    ]
    srates = []
    for alpha in alphas:
        Dtilde = pp.combinedProximity(
                            alpha=alpha)
        p_a = pfm.Performance(Dtilde,mu)
        SR = p_a.successRateKNN(kmax)
        srates.append((alpha,SR[0]))
        RC = p_a.recallKNN(kmax)
        PR = p_a.precisionKNN(kmax)
        y_vals.append([ 
            [SR, RC, PR],
            [p_a.Fmeasure(beta=1.0,
                          PR=PR,RC=RC), 
             p_a.Fmeasure(beta=2.0,
                          PR=PR,RC=RC),
             p_a.ANMRR(kmax)]
        ])            
        labels.append('alpha='+str(alpha))
    pfm.plotGridOfFunctionBags(x_vals,y_vals,
        labels=labels,lrow=lrow,lcol=lcol,
        title= title, titles=titles,
        win_title=win_title)
    return srates
class PR:
    P = 159 # number of "training persons"
            # = number of remaining = "testing" persons
    l = 5   # number of images per person

    m = 56  # number of pixels in image column
    n = 46  # number of pixels in image row

    a = 7   # height of left upper block in DFT domain
    b = 9   # width of left upper block in DFT domain

    c = 7   # height of right upper block in DFT domain
    d = 8   # width of right upper block in DFT domain
from numpy.fft import fft2,ifft2
class Spectrum:
    from numpy.fft import fft2,ifft2
    def __init__(self,X):
        self.X2D = X
        self.X3D = X.reshape(PR.m,PR.n,X.shape[1])
        self.Y3D = fft2(self.X3D,axes=(0,1),norm='ortho')
    #def view2D(self):
        #self.Y2D = self.Y3D.reshape(*self.X2D.shape)
        #return self.Y2D
    def extract(self,abcd=(PR.a,PR.b,PR.c,PR.d)):
        self.abcd = abcd
        a,b,c,d = abcd
        L = self.Y3D.shape[2]
        Y_LU = self.Y3D[:a,:b,:].reshape(a*b,L)
        Y_RU = self.Y3D[:c,-d:,:].reshape(c*d,L)
        self.channels = np.concatenate([
            Y_LU.real, Y_LU.imag,
            Y_RU.real, Y_RU.imag],axis=0)
        return self.channels
    def reconstruct(self,C):
        a,b,c,d = self.abcd
        if C.ndim!=2:
            e = 'Reconstruction only for 2D arrays'
            raise Exception(e)
        if (2*(a*b+c*d))!=C.shape[0]:
            e = 'Channels corners incompatible'+\
                  ' with channel number'
            raise Exception(e)    
        ab = a*b; ab2 = 2*ab; cd = c*d
        K,L = C.shape
        m,n,L_all = self.Y3D.shape
        Y_LU = C[:ab,:]+1j*C[ab:ab2,:]
        Y_RU = C[ab2:ab2+cd,:]+\
               1j*C[ab2+cd:ab2+2*cd,:]        
        Yrec = np.zeros((m,n,L),dtype=complex)
        Yrec[:a,:b,:] = Y_LU.reshape(a,b,L)
        Yrec[:c,-d:,:] = Y_RU.reshape(c,d,L)
        Y_LUc = Y_LU.conj().reshape(a,b,L)
        Yrec[-(a-1):,-(b-1):,:] = np.flip(np.flip(\
                   Y_LUc[1:a,1:b,:],axis=1),axis=0)
        Y_RUc = Y_RU.conj().reshape(c,d,L)
        Yrec[-(c-1):,1:d,:] = np.flip(np.flip(\
           Y_RUc[1:c,-(d-1):,:],axis=1),axis=0)
        Yrec[-(a-1):,0,:] =\
                  np.flip(Y_LUc[1:a,0,:],axis=0)
        Xrec = ifft2(Yrec,axes=(0,1),norm='ortho')
        return np.abs(Xrec)
def tasks_B_1(path):
    # TASK B-1.1
    print('\n'+'+++TASK B-1.1'*4)
    beg = clock()
    X_in_train,mu_train = facesToMatrix(
                  path+'/FACES-PNG/',mu_yes=True)
    end = clock()
    print('Reading of '+str(X_in_train.shape[1])+\
    ' faces in {0:.1f}[ms]'.format((end-beg)*1e3))
    beg = clock()
    dft_train = Spectrum(X_in_train)
    end = clock()
    print('DFT-2D processing in {0:.1f}[ms]'.\
                        format((end-beg)*1e3))
    Y3D_real = dft_train.Y3D[:,:,:20].real
    Y3D_imag = dft_train.Y3D[:,:,:20].imag
    Y3D_abs = np.abs(dft_train.Y3D[:,:,:20])
    Y3D_list = tolist(Y3D_real,axis=-1)
    Y3D_list.extend(tolist(Y3D_imag,axis=-1))
    Y3D_list.extend(tolist(Y3D_abs,axis=-1))
    imsave(path+'/dft-real-imag-abs.jpg',
    formImageGrid(Y3D_list,gridx=20,scale='log'))
    print('Grid of DFT images is saved to '+\
          'data/dft-real-imag-abs.jpg')
    XYR_list = tolist(dft_train.X3D[:,:,:20],
                      axis=-1)
    Y3D_abs_log = np.log10(np.abs(
             dft_train.Y3D[:,:,:20])+1e-100)
    XYR_list.extend(tolist(Y3D_abs_log,
                    axis=-1))
    rec_X3D = ifft2(dft_train.Y3D,axes=(0,1),
                    norm='ortho')
    rec_X3D_mag = np.abs(rec_X3D[:,:,:20])
    XYR_list.extend(tolist(rec_X3D_mag,
                    axis=-1))
    imsave(path+'/img-dft-rec.jpg',
    formImageGrid(XYR_list,gridx=20)) 
    print('Reconstruction error '+\
          '||X-rec_X||:{:.3g}'.format(
    norm(dft_train.X3D-rec_X3D)/rec_X3D.size))
    print('Grid of original, log|DFT|,'+\
          ' and reconstructed is saved to'+\
          ' data/img-dft-rec.jpg')
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK B-1.1: TO DO for two points
    F = dft_train.extract()
    ### YOUR CODE HERE
    #   ...
    ###
    print('-'*40)    
    # TASK B-1.2
    print('\n'+'+++TASK B-1.2'*4)

    from sys import argv
    model = 'LDA'; regu = False
    q = 180; m = 48
    cntr = False; e = -14; eps = 10.**e
    n = len(argv)
    if n>1: model = argv[1]
    if n>2: 
        d = eval('dict('+argv[2]+')')
        if 'q' in d: q = d['q']; regu = False
        if 'm' in d: m = d['m']
        if 'c' in d: cntr = d['c']
        if 'e' in d: e = d['e']
        eps = 10.**e; regu = True

    print('model:',model,'r=',regu,'q=',q,
                         'm=',m,'c=',cntr)    
    beg = clock()
    data_model = tss4lda(
         dft_train.channels,mu_train,
         m=m,model=model,q=q,eps=eps,
         regularize=regu,centre=cntr) 
    end = clock()
    print('LDA build: {0:.1f}[ms]'\
          .format((end-beg)*1e3))
    print('-'*40)    
    X_in_test,mu_test =  facesToMatrix(
         path+'/FACES-PNG/',start=3,mu_yes=True)
    dft_test = Spectrum(X_in_test)
    X_test= dft_test.extract()
    F = ldaCoding(data_model,X_test,
                  grand_centre=cntr)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    plotPerformanceMeasures(F,mu_test,
    alphas=alphas,
    title='Performance measures for default '+\
          'configuration of LDA classifier,\n'+\
    'the combined proximities get weights from '+\
                                  str(alphas),
    win_title= 'TASK B-1.2 for maximal k_NN buffer')
    print('Performance measures SR,RC,PR,'+\
    'F_1,F_2,ANMRR are computed and plotted'+\
    '\nfor combined proximities with '+\
    'alpha in '+str(alphas)+\
    '\nwhile model parameters are default.')
    kNN = 10
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK B-1.2: TO DO for two points
    ### YOUR CODE HERE
    #   ...
    ###
    print('The same but for k-NN buffer with '+\
          '{:d} elements.'.format(kNN))
    print('='*60)        
    return dft_train,mu_train,dft_test,mu_test      
def matrixToGrayImage(A,scale='affine'):
    As = np.float64(A).copy(); As -= np.min(As)
    if scale.startswith('log'):
            As = np.log10(As+1.0)
            abar = np.mean(As); astd = np.std(As)
            abar0 = max(abar-3*astd,0.)
            abar1 = abar+3*astd
            As[As<abar0] = abar0; As[As>abar1] = abar1
            As -= np.min(As)
    As *= 255/np.max(As)
    return np.uint8(As+.5)
def formImageGrid(Flist,gridx=7,scale='normal'):
    h,w =  Flist[0].shape
    gridy = len(Flist)//gridx
    if gridy*gridx<len(Flist): gridy = gridy+1
    G = np.zeros((h*gridy,w*gridx),dtype=np.uint8)
    k = 0    
    for m in range(gridy):
        for n in range(gridx):
            G[m*h:(m+1)*h,n*w:(n+1)*w] =\
            matrixToGrayImage(Flist[k],scale=scale)
            k = k+1
            if k>=len(Flist): break
        if k>=len(Flist): break
    return G
def tofirst(li,pos):
    if pos<0: pos = len(li)+pos
    li2 = [li[pos]];     li2.extend(li[:pos])
    li2.extend(li[(pos+1):])
    return li2
def tolist(F,axis=-1):
    if axis==0: return list(F)
    shape = tofirst(F.shape,axis)
    strides = tofirst(F.strides,axis)
    Ft = as_strided(F,shape=shape,strides=strides)
    return list(Ft)
def tss4lda(X,mu,m=48,q=180,model='LDA',
            regularize=False,centre=True,eps=1e-14):
            
    pp = utl.ProximityPlus(X.T,mu=mu,
             grand_centre=centre,
             class_centre=True,
             class_means_centre=True)
    N = pp.class_centered
    D = pp.class_means_centered
    mux = pp.grand_mean
    if model=='LDA': N,D = D,N    
    A = get_white_projection_on_DSS(D.T,q,regularize,
                                             eps=eps)                                            
    Y = np.dot(A,N.T) 
        # data whitening
    U_Nm,sigmas_Nm = get_subspace_of_Y(Y,m,model)                                                   
    W = np.dot(A.T,U_Nm) 
        # matrix of discriminant transform                                            
    return mux,W,sigmas_Nm                                    
def regularize_D(sigmas_D,q_D,n,eps=1e-5):
    epsilon = np.sum(sigmas_D)*eps/(n-q_D)
    sigmas = epsilon*np.ones(n)
    sigmas[:q_D] = sigmas_D[:q_D]                                   
    return sigmas
def get_white_projection_on_DSS(D,q,regularize,eps=1e-5):
    n = D.shape[0]
    U_D,sigmas_D,V_Dt = svd(D)                                           
    q_D = real_rank_by_svalues(sigmas_D)
    if q_D==0: raise Exception('Single point data')
    if regularize:
        if q_D<n: sigmas_D = regularize_D(sigmas_D,
                                   q_D,n,eps=eps)
        q = n
    else:
        if q<=0 or q>q_D: q = q_D
    sigmas_Dq = sigmas_D[:q].copy()
    U_Dq = U_D[:,:q].copy()
    sigmas_Dq_inv = 1./sigmas_Dq
    A = np.dot(np.diag(sigmas_Dq_inv),U_Dq.T)
    
    #print('actual rank of D(enominator:',q_D)
    #print('q dimension:',q)
    return A

def real_rank_by_svalues(sigmas,fraction=1-1e-10):
    if fraction>=1.0: fraction = 1-1e-10
    csum = np.cumsum(sigmas)
    threshold = csum[-1]*fraction
    ind, = np.where(csum>threshold)
    rank = ind[0]+1
    return rank
def get_subspace_of_Y(Y,m,model):
    U_N,sigmas_N,V_Nt = svd(Y)
                                           
    q_N = real_rank_by_svalues(sigmas_N)
    if m<=0 or m>q_N: m = q_N    
    if model=='DLDA':
        U_Nm = U_N[:,q_N-m:q_N]
        sigmas_Nm = sigmas_N[q_N-m:q_N]
    else:
        U_Nm = U_N[:,:m]
        sigmas_Nm = sigmas_N[:m]
        
    #print('actual rank of N(ominator:',q_N)
    #print('feature vector dimension m:',m)
    return U_Nm,sigmas_Nm
def ldaCoding(data_model,X,grand_centre=False):
    mux,W,sigmas = data_model
    if grand_centre:
        X -= as_strided(mux,shape=X.shape,
             strides=(mux.strides[0],0))
    Y = np.dot(X.T,W)
    return Y
def tasks_B_2(dft_train,mu_train,dft_test,mu_test):
    # TASK B-2.1
    print('\n'+'+++TASK B-2.1'*4)
    data_model = tss4lda(dft_train.channels,
              mu_train,m=48,model='LDA',q=180,
              regularize=False,centre=False)
    mux,W,sigmas = data_model              
    mstr = 'M1r0c0m48q180'              
    FisherFaces(dft_train,W,mstr)
    Y = ldaCoding(data_model,dft_test.channels,
                  grand_centre=False)
    plotPerformanceMeasuresAndPrintSrates(Y,
           mu_test,mstr,task='B-2.1',kNN=10)    
    print('-'*40)
    # TASK B-2.2
    print('\n'+'+++TASK B-2.2'*4)
    kmax=1; alpha=0.0; centre = False
    data_model,q = findBestDataModel('optimize SR',
              dft_train.channels,mu_train,
              dft_test.channels,mu_test,
              m=48,model='LDA',kNN=kmax,alpha=alpha,
              regularize=False,centre=centre)
    mux,W,sigmas = data_model              
    mstr = 'M0r0c0m48q'+str(q)              
    FisherFaces(dft_train,W,mstr)
    Y = ldaCoding(data_model,dft_test.channels,
                  grand_centre=centre)    
    plotPerformanceMeasuresAndPrintSrates(Y,
           mu_test,mstr,kNN=10,
           task='B-2.2 alpha={:.2f} kNN={:d}'\
                .format(alpha,kmax))    
    print('-'*40)
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK B-2.3: TO DO for one point
    print('\n'+'+++TASK B-2.3'*4)
    ### YOUR CODE HERE
    #   ...
    ###
    print('-'*40)
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK B-2.4: TO DO for two points
    print('\n'+'+++TASK B-2.4'*4)    
    ### YOUR CODE HERE
    #   ...
    ###
    print('-'*40)
    # TO DO -- TO DO -- TO DO -- TO DO -- TO DO
    # TASK B-2.5: TO DO for one point
    print('\n'+'+++TASK B-2.5'*4)
    ### YOUR CODE HERE
    #   ...
    ###
    print('='*60)
def FisherFaces(dft_train,W,mstr):
    Wr = dft_train.reconstruct(W)
    Wr_list = tolist(Wr,axis=2)
    imsave(path+'/fisher-faces-'+mstr+'.jpg',
          formImageGrid(Wr_list,gridx=16))
    print('Fisher faces of model '+mstr+' to',
          'data/fisher-faces-'+mstr+'.jpg')                 
def plotPerformanceMeasuresAndPrintSrates(
                 Y,mu,mstr,task='',kNN=10):
    alphas = [0,.25,.5,.75,1]
    srates = plotPerformanceMeasures(Y,mu,
        alphas=alphas,kNN=kNN,lrow=1,lcol=0,
        title='Performance measures for model '+\
        mstr+'\nthe combined proximities'+\
        ' get weights from '+str(alphas),
        win_title= 'TASK '+task)
    print('Success rate for model',mstr)
    txt = ' '*13+'{:.3f} at alpha = {:.2f}'
    for alpha,sr in srates:
        print(txt.format(sr,alpha))
def getMeasureValue(todo,pp,mu,alpha,kmax=10):
    Dtilde = pp.combinedProximity(alpha=alpha)
    pf = pfm.Performance(Dtilde,mu)
    if 'SR' in todo:
        val = np.sum(pf.successRateKNN(kmax))
    elif 'RC' in todo:
        val = np.sum(pf.recallKNN(kmax))
    elif 'PR' in todo:
        val = np.sum(pf.precisionKNN(kmax))
    elif 'ANMRR' in todo:
        val = np.sum(pf.ANMRR(kmax))
    return val
def getBestModelIndex(todo,dmodels):
    irange = range(len(dmodels))
    if 'ANMRR' in todo:
        best = min(irange,
               key=lambda i:dmodels[i][2])
    else:
        best = max(irange,
               key=lambda i:dmodels[i][2])
    return best
def findBestDataModel(todo,
       X_train,mu_train,X_test,mu_test,
       m=48,model='LDA',kNN=10,alpha=0.5,
       regularize=False,centre=False):
    if regularize:
        arange = range(-14,+3,+2)
    else:
        N,L = X_train.shape
        if model=='DLDA': qmax = min(PR.P,N)
        else: qmax =  min(L,N)
        arange = range(92,qmax,8)
    dmodels = []
    print('Looking for the best model in time ...',
          end=' ',flush=True)
    beg = clock()
    for eq in arange:
        data_model = tss4lda(X_train,mu_train,
             m=m,model=model,
             regularize=regularize,
             q=-1 if regularize else eq,
             eps=10.**eq if regularize else 1.,
             centre=centre)
        Y = ldaCoding(data_model,X_test,
                      grand_centre=centre)    
        pp = utl.ProximityPlus(Y,mu_test)
        val = getMeasureValue(todo,pp,mu_test,
                              alpha,kmax=kNN)
        dmodels.append((eq,data_model,val))
    best = getBestModelIndex(todo,dmodels)
    eq_best,dmodel_best,val_best = dmodels[best]
    dt = clock()-beg
    print('{:.1f}[sec]'.format(dt),flush=True)
    return dmodel_best,eq_best
if __name__=='__main__':
    path = './data'
    tasks_A_1(path)
    tasks_A_2(path)
    #Z,F,mu = tasks_A_3(path)
    #tasks_A_4(Z,F,mu,path)
    #tasks_B_2(*tasks_B_1(path))