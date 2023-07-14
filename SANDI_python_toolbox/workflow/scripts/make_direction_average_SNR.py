import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
# Calculate the direction-average of the data

# Original Author:
# Dr. Marco Palombo
# Cardiff University Brain Research Imaging Centre (CUBRIC)
# Cardiff University, UK
# 8th December 2021
# Email: palombom@cardiff.ac.uk

# Ported into python by Bradley Karat
# University of Western Ontario
# 20th May 2023
# Email: bkarat@uwo.ca

tic = time.time()

f = open(snakemake.log.log, "a")
f.write("Computing the Spherical Mean for dataset")
f.close()

# Load the imaging data

tmp = nib.load(snakemake.input.mask)
mask = np.double(tmp.get_fdata()).flatten(order='F')

tmp = nib.load(snakemake.input.noisemap)
tmp_img = np.double(tmp.get_fdata()).flatten(order='F')
sigma_mppca = np.transpose(tmp_img[mask==1])

tmp = nib.load(snakemake.input.dwi)
I = np.double(np.abs(tmp.get_fdata()))
slope = tmp.header['scl_slope']
intercept = tmp.header['scl_inter']

#I = np.multiply(I,slope+1) + intercept+1; #Nibabel handles scaling internally
#tmp.img = []

[sx, sy, sz, vols] = I.shape
FWHM = np.double(snakemake.params.FWHM[0])
SNR = np.double(snakemake.params.SNR[0])

sigma = FWHM/(2*np.sqrt(2*np.log(2)))

for i in range(vols):
    I[:,:,:,i] = gaussian_filter(np.squeeze(I[:,:,:,i]), sigma, mode='nearest')
    
# Load bvals and bvecs
bvecs = np.loadtxt(snakemake.input.bvec)

bvals = np.loadtxt(snakemake.input.bval)
bvals = np.multiply(np.round(bvals/100),100)

bunique = np.unique(bvals)

Save = np.zeros((sx, sy, sz, len(bunique)))
S0mean = np.nanmean(np.double(I[:,:,:,bvals<=100]),axis=3)
S0mean_vec = S0mean.flatten()
S0mean_vec = S0mean_vec[mask==1]

# Identify b-shells and direction-average per shell
estimated_sigma = np.zeros((len(bunique),S0mean_vec))
Ndirs_per_shell = np.zeros((1, len(bunique)))
for i in range len(bunique):
    
    Ndirs_per_shell[0,i] = np.sum(bvals==bunique[i])    

    if i==0:
        Save(:,:,:,i) = S0mean;            
        term1 = np.divide(1,SNR)
        sigma_mppca = np.multiply(term1,np.transpose(S0mean_vec))
        estimated_sigma[i,:] = sigma_mppca

    elif i>0:
        
        dirs = np.transpose(bvecs[:,bvals==bunique[i]])
        
        Ndir = np.sum(bvals==bunique[i])
        
        if Ndir<=15:             
            order = 2
        elif Ndir>15 and Ndir<=28:
            order = 4
        elif Ndir>28 and Ndir<=45:  
            order = 6
        elif Ndir>45:              
            order = 8


        f = open(snakemake.log.log, "a")
        f.write(f"\nFitting SH to the per-shell data with order l= {str(order)} -shell {str(i-1)} of {str(len(bunique)-1)} -directions = {str(np.sum(bvals==bunique[i]))}")
        f.close()

        y_tmp = I[:,:,:,bvals==bunique[i]]
        y_tmp = np.reshape(y_tmp,[sx*sy*sz, np.shape(y_tmp)[3]],order='F')
        y_tmp = np.transpose(y_tmp)
        indhold = mask==1
        y = y_tmp[:,indhold]
        [Save_tmp, sigma_tmp] = SphericalMeanFromSH(dirs, y, order)
        
        estimated_sigma[i,:] = sigma_tmp                         #np.concatenate((estimated_sigma, sigma_tmp))

        if not snakemake.params.diravg:
            f = open(snakemake.log.log, "a")
            f.write(f"\nCalculating powder average signal as aritmetic mean over the directions. The SH fit is only used to estimate the variance of the gaussian noise")
            f.close()
            Save[:,:,:,i] = np.nanmean(I[:,:,:,bvals==bunique[i]], axis=3)
        else:
            tmp_img = np.zeros((sx*sy*sz, 1))
            tmp_img[mask==1] = Save_tmp
            Save[:,:,:,i] = np.reshape(tmp_img,[sx,sy,sz])

for mm in range(Save.shape[-1]):
    Save[:,:,:,mm] = np.divide(Save[:,:,:,mm],S0mean)

# Save the direction-averaged data in NIFTI
tmpsave = nib.Nifti1Image(Save,tmp.affine) # We will use the normalized spherical mean signal
nib.save(tmpsave,snakemake.output.direction_avg)

sigma_SHresiduals = np.nanmean(estimated_sigma,axis=0)
noisemap = np.zeros((sx*sy*sz, 1))
np.place(noisemap, mask, sigma_SHresiduals)
tmpsaveimg = np.reshape(noisemap,[sx,sy,sz],order='F') # We will use the normalized spherical mean signal

tmpsave = nib.Nifti1Image(tmpsaveimg,tmp.affine) # We will use the normalized spherical mean signal
nib.save(tmpsave,snakemake.output.noisemap)

noisemap_mppca = normalize_noisemap(snakemake.input.noisemap, snakemake.input.dwi, snakemake.input.mask, snakemake.input.bval) # Load the noisemap previously obtained by MP-PCA denoising and normalizes it by dividing it for the b=0 image.
sigma_mppca = noisemap_mppca.flatten()
sigma_mppca = sigma_mppca[sigma_mppca>0]
sigma_mppca[sigma_mppca<0] = np.nan
sigma_mppca[sigma_mppca>1] = np.nan
tmphold = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_mppca.npy')
tmpnew = np.append(tmphold,sigma_mppca,axis=0)
np.save(f'{snakemake.params.tmpdir}/hold_noisemap_norm_mppca.npy',tmpnew)

noisemap_SHresiduals = normalize_noisemap(snakemake.output.noisemap, snakemake.input.dwi, snakemake.input.mask, snakemake.input.bval) # Load the noisemap previously obtained by MP-PCA denoising and normalizes it by dividing it for the b=0 image.
sigma_SHresiduals = noisemap_SHresiduals[:]
sigma_SHresiduals = sigma_SHresiduals[sigma_SHresiduals>0]
sigma_SHresiduals[sigma_SHresiduals<0] = np.nan
sigma_SHresiduals[sigma_SHresiduals>1] = np.nan
tmphold = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_SHresiduals.npy')
tmpnew = np.append(tmphold,sigma_SHresiduals,axis=0)
np.save(f'{snakemake.params.tmpdir}/hold_noisemap_norm_SHresiduals.npy',tmpnew)

if snakemake.params.QC:
    
    slices = Save.shape[2]
    slice_to_show = int(np.round(slices/2))
    img_to_show = np.squeeze(Save[:,:,slice_to_show,:])
    
    numofimg = img_to_show.shape[2]

    fig, axs = plt.subplots(nrows=round(numofimg/2)+1,ncols=round(numofimg/2)+1,figsize=(15,15))

    axs = axs.ravel()
    for ii,ax in enumerate(axs):
        if ii>=numofimg:
            fig.delaxes(ax)
        else:
            ax.imshow(img_to_show[:,:,ii],cmap='gray',vmin=0.01,vmax=0.65)

    txt=f"The plot shows the direction averaged signal for each b value, normalized by the b=0 image. \nThe dataset has {len(bunique)} b values: {bunique} in s/mm^2, with {Ndirs_per_shell} number of directions. \nThe Delta is {np.double(snakemake.params.Delta)} ms; the smalldelta is {np.double(snakemake.params.smalldelta)} ms; \nThe diffusion time is {np.double(snakemake.params.Delta) - (np.double(snakemake.params.smalldelta)/3)} ms."
    plt.gcf().text(0.2,0.5, txt)#ha='right',va='bottom')
    plt.savefig(snakemake.output.QC)
else:
    fig = plt.figure()
    txt = 'no QC: Use the --use_QC flag to generate figures'
    plt.gcf().text(0.2,0.5, txt)
    plt.savefig(snakemake.output.QC)




toc = time.time()
tottime = (toc - tic)
f = open(snakemake.log.log, "a")
f.write(f'\nDONE - Spherical mean computed in {round(tottime)} sec.')
f.close()

