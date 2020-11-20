
from numpy import mean,cov,cumsum,dot,linalg,size,argsort,shape,reshape,concatenate,hstack,vstack,array,poly1d,polyfit,linspace
from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator,savefig,text,bar,close,xlabel,ylabel
from skimage.measure import compare_ssim

pic_orig = imread('profilepic256x256.png') # must be a 256x256 image i.e. (16x16)x(16x16)
pic_orig = mean(pic_orig,2)
#pic_orig_r = pic_orig[:,:,0] # to be done: PCA reconstruction of colour image
#pic_orig_g = pic_orig[:,:,1]
#pic_orig_b = pic_orig[:,:,2]

# split the image into 256 patches of size 16x16
patches=[]
scores=[]
for i in range (0, 256, 16):
    for j in range (0, 256, 16):
        patches.append(pic_orig[i:i+16,j:j+16])

for k in range(0,16,1): # there are a total of 16 principal components for each patch
    patch_PC=[]
    patches_reconstructed=[]
    for i in range(256):
        M = (patches[i]-mean(patches[i].T,axis=1)).T   # normalize along columns
        eig_val,eig_vec = linalg.eig(cov(M))
        idx = argsort(eig_val)                         # sort the eigenvalues
        idx = idx[::-1]
        eig_vec = eig_vec[:,idx]                       # sort the eigenvalues
        eig_val = eig_val[idx]                         # sorting eigenvalues
        eig_vecs = eig_vec[:,range(k)]                 # only select k PCs for image reconstruction
        score = dot(eig_vecs.T,M)
        patches_reconstructed.append(dot(eig_vecs,score).T+mean(patches[i],axis=0)) # reconstruct image
    
    # plot reconstructed images (i.e. with multiple PCs)
    rows=[]
    patches_reconstructed_img=[]
    for i in range (0, 256, 16):
        rows.append(patches_reconstructed[i:i+16])
        row = hstack(patches_reconstructed[i:i+16])
        if i==0:
            patches_reconstructed_img = row
        else:
            patches_reconstructed_img = vstack((patches_reconstructed_img,row))
    
    # Calulate the Structural Similarity Index (SSIM) between the two images
    score,_ = compare_ssim(pic_orig, patches_reconstructed_img, full=True)
    scores.append(score)

    ks=str(list(range(0, k+1)))
    imshow(patches_reconstructed_img)
    title('k = '+ ks)
    text(3, 10, "SSIM: {}".format(score), style='italic', fontsize=8, 
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})

    gray()
    savefig('output/k='+ks+'.png')
    close()

# plot similarity scores
figure(1)
x = array(list(range(0, k+1)))
y = array(scores)
bar(x, y, align='center', alpha=0.5)
title('SSIM by number of PCs')
xlabel('number of Principal Components (PCs) used')
ylabel('Structural Similarity Index (SSIM)')
savefig('output/SSIM_by_PCs.png')
close()
