import nibabel as nib
from skimage import morphology
import numpy as np

def normalize_noisemap(noisemap_filename, data_filename, mask_filename, bvalues_filename):

    tmp = nib.load(noisemap_filename)
    noisemap = np.double(tmp.get_fdata())

    tmp = nib.load(mask_filename)
    mask = np.double(tmp.get_fdata())

    kernel = morphology.cube(9)
    mask = morphology.erosion(mask,kernel)

    tmp = nib.load(data_filename)
    dwi = np.double(tmp.get_fdata())

    bvals = np.loadtxt(bvalues_filename)

    b0mean = np.nanmean(dwi[:,:,:,bvals<100], 3)
    term1 = np.divide(noisemap,b0mean)
    noisemap = np.multiply(term1,mask)

    return noisemap