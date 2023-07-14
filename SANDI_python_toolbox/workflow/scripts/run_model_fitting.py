import numpy as np
import nibabel as nib
import pickle
from apply_RF_python import apply_RF_python
from apply_MLP_python import apply_MLP_python
import subprocess

# Main script to fit the SANDI model using Random
# Forest (RF) and/or multi-layers perceptron (MLP)

# Author:
# Dr. Marco Palombo
# Cardiff University Brain Research Imaging Centre (CUBRIC)
# Cardiff University, UK
# 8th December 2021
# Email: palombom@cardiff.ac.uk

# Ported into python by Bradley Karat
# University of Western Ontario
# 20th June 2023
# Email: bkarat@uwo.ca

np.random.seed(1)

# delta = snakemake.params.Delta
# smalldel = snakemake.params.smalldelta
# sigma_mppca = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_mppca.npy')
# sigma_SHresiduals = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_SHresiduals.npy')
# Dsoma = snakemake.params.Dsoma
# Din_UB = snakemake.params.Din_UB
# use_Rsoma_UB = snakemake.params.Rsoma_UB
# De_UB = snakemake.params.De_UB
# FittingMethod = snakemake.params.FittingMethod
# Nset = snakemake.params.Nset
# method = snakemake.params.method

mask_data = snakemake.input.mask
MLmodel = str(snakemake.params.MLmodel[0])
QC = snakemake.params.QC
method = snakemake.params.method

Mdl = snakemake.input.model
with open(Mdl, "rb") as fp:
    # Load the model dictionary from the file
    trainedML = pickle.load(fp)

modelinf = snakemake.input.modelinfo
with open(modelinf, "rb") as fp:
    # Load the model information dictionary from the file
    modelinfo = pickle.load(fp)

# Load direction-averaging with gaussian smoothing given the provided FHWM
img_data = snakemake.input.direction_avg # data to process: direction-averaged signals for each subject

# Load data

tmpmask = nib.load(mask_data)
mask = np.double(tmpmask.get_fdata())

tmp = nib.load(img_data)
tmpimg = tmp.get_fdata()

I = np.double(tmpimg)
[sx, sy, sz, vol] = I.shape

f = open(snakemake.log.log, "a")
f.write(f"Data {img_data} loaded:")
f.write(f"\nMatrix size = {sx} x {sy} x {sz}")
f.write(f"\nVolumes = {vol}")
f.close()

delta = float(snakemake.params.Delta[0])
smalldel = float(snakemake.params.smalldelta[0])
bvals = snakemake.input.bvals
bvals = np.loadtxt(bvals)

f = open(snakemake.log.log, "a")
f.write(f"\nProtocol loaded:")
f.write(f"\nGradient pulse duration = {smalldel} ms")
f.write(f"\nGradient pulse seperation = {delta} ms")
f.write(f"\nDiffusion time = {delta - smalldel/3} ms")
f.write(f"\nb-values = {np.unique(bvals)} s/mm^2")
f.write(f"\n# {np.sum(bvals>=100)} = b shells")
f.close()

# Prepare ROI for fitting
ROI = np.reshape(I, [sx*sy*sz,vol])
m = np.reshape(mask, [sx*sy*sz])
signal = (ROI[m==1,:])

# Remove nan or inf and impose that the normalised signal is >= 0
signal[np.isnan(signal)] = 0
signal[np.isinf(signal)] = 0
signal[signal<0] = 0

noMLdebias = snakemake.params.MLdebias

# Fitting the model to the data using pretrained models
if MLmodel == 'RF':
    f = open(snakemake.log.log, "a")
    f.write(f"\nFitting using a Random Forest regressor")
    f.close()  

    f = open(snakemake.log.log, "a")
    f.write(f"\nApplying the Random Forest...")
    f.close()  

    mpgMean = apply_RF_python(signal, trainedML, noMLdebias,snakemake.log.log)

elif MLmodel == 'MLP':

    f = open(snakemake.log.log, "a")
    f.write(f"\nFitting using a MLP regressor")
    f.close()  

    f = open(snakemake.log.log, "a")
    f.write(f"\nApplying the MLP...")
    f.close()  

    mpgMean = apply_MLP_python(signal, trainedML, noMLdebias,method,snakemake.log.log)


# Calculate and save SANDI parametric maps
paramsrange = modelinfo['paramsrange']
names = ['fneurite', 'fsoma', 'Din', 'Rsoma', 'De', 'fextra', 'Rsoma_Low_fsoma_Filtered', 'Din_Low_fsoma_Filtered']
#bounds = np.array(([0,0.7],[0,0.7],[paramsrange[2:5,:]],[0,1]))

mpgMean = np.abs(mpgMean)

fneu = mpgMean[:,0]

fsom = mpgMean[:,1]

fe = 1 - fneu - fsom

fneurite = fneu / (fneu + fsom + fe)
fsoma = fsom / (fneu + fsom + fe)
fextra = fe / (fneu + fsom + fe)

f = open(snakemake.log.log, "a")
f.write(f"\nSaving SANDI parameter maps")
f.close()  

for i,outfile in enumerate(snakemake.output.maps):

    itmp = np.zeros((sx*sy*sz))

    if i==0:
        itmp[m==1] = fneurite
    elif i==1:
        itmp[m==1] = fsoma
    elif i==mpgMean.shape[1]:
        itmp[m==1] = fextra
    elif i==mpgMean.shape[1]+1:
        Rsoma_tmp = mpgMean[:,3]
        Rsoma_tmp[fsoma<=0.15] = 0
        itmp[m==1] = Rsoma_tmp
    elif i==mpgMean.shape[1]+2:
        Din_tmp = mpgMean[:,2]
        Din_tmp[fneurite<0.10] = 0
        itmp[m==1] = Din_tmp
    else:
        mpgMean[mpgMean[:,i]<0,i] = 0
        itmp[m==1] = mpgMean[:,i]

    itmp = np.reshape(itmp,[sx,sy,sz])

    imghold = nib.Nifti1Image(itmp,tmpmask.affine)
    nib.save(imghold,outfile)

tmpdir = snakemake.params.tmpdir
subprocess.run(['rm','-r', tmpdir])


# for i in range(mpgMean.shape[1] + 2):

#     itmp = np.zeros((sx*sy*sz,1))
#     ind = []

#     for k in range(len(names)):
#         if names[k] in snakemake.output.maps[0]:
#             ind.append(k)
    
#     if len(ind) > 1:
#         lenhold = []
#         for j in range(len(ind)):
#             lenhold.append(len(names[ind[j]]))
#         numind = ind[np.argmax(lenhold)]
#     else:
#         numind = ind[0]

#     if numind==0:
#         itmp[mask==1] = fneurite
#     elif numind==1:
#         itmp[mask==1] = fsoma
#     elif numind==mpgMean.shape[1]:
#         itmp[mask==1] = fextra
#     elif numind==mpgMean.shape[1]+1:
#         Rsoma_tmp = mpgMean[:,3]
#         Rsoma_tmp[fsoma<=0.15] = 0
#         itmp[mask==1] = Rsoma_tmp
#     elif numind==mpgMean.shape[1]+2:
#         Din_tmp = mpgMean[:,2]
#         Din_tmp[fneurite<0.10] = 0
#         itmp[mask==1] = Din_tmp
#     else:
#         mpgMean[mpgMean[:,numind]<0,numind] = 0
#         itmp[mask==1] = mpgMean[:,numind]

#     itmp = np.reshape(itmp,[sx,sy,sz])

#     imghold = nib.Nifti1Image(itmp,tmpmask.affine)
#     nib.save(imghold,snakemake.output.maps)



#if QC:


# try
#     if diagnostics==1
#         r = SANDIinput.report(SANDIinput.subj_id, SANDIinput.ses_id).r;
#     end

#     for i=1:size(mpgMean,2)+3

#         itmp = zeros(sx*sy*sz,1);

#         if i==1
#             itmp(mask==1) = fneurite;
#         elseif i==2
#             itmp(mask==1) = fsoma;
#         elseif i==size(mpgMean,2)+1
#             itmp(mask==1) = fextra;
#         elseif i==size(mpgMean,2)+2
#             Rsoma_tmp = mpgMean(:,4);
#             Rsoma_tmp(fsoma<=0.15) = 0;
#             itmp(mask==1) = Rsoma_tmp;
#         elseif i==size(mpgMean,2)+3
#             Din_tmp = mpgMean(:,3);
#             Din_tmp(fneurite<0.10) = 0;
#             itmp(mask==1) = Din_tmp;
#         else
#             mpgMean(mpgMean(:,i)<0,i) = 0;
#             itmp(mask==1) = mpgMean(:,i);
#         end

#         itmp = reshape(itmp,[sx sy sz]);

#         if diagnostics==1
#             if i<=size(mpgMean,2)+1
#                 h = figure('Name',['SANDI ' names{i} ' map']); hold on
#                 tmp = imtile(itmp);
#                 imshow(tmp,bounds(i,:)), title(names{i}), colorbar, colormap parula

#                 T = getframe(h);
#                 imwrite(T.cdata, fullfile(output_folder , 'SANDIreport', ['SANDI ' names{i} ' map.tiff']))

#                 r.section(['SANDI ' names{i} ' map']);
#                 r.add_text(['The plot shows the SANDI ' names{i} ' map.']);
#                 r.add_figure(gcf,['SANDI ' names{i} ' map. Units are um for maps indicating radius and um^2/ms for maps indicating diffusivities.'],'left');
#                 r.end_section();
                

#                 savefig(h, fullfile(output_folder, 'SANDIreport', ['SANDI ' names{i} ' map.fig']));
#                 close(h);
#             end
#         end

#         nifti_struct.img = itmp;
#         nifti_struct.hdr.dime.dim(5) = size(nifti_struct.img,4);
#         if size(nifti_struct.img,4)==1
#             nifti_struct.hdr.dime.dim(1) = 3;
#         else
#             nifti_struct.hdr.dime.dim(1) = 4;
#         end
#         nifti_struct.hdr.dime.datatype = 16;
#         nifti_struct.hdr.dime.bitpix = 32;

#         save_untouch_nii(nifti_struct,fullfile(output_folder, ['SANDI-fit_' names{i} '.nii.gz']));
#         disp(['  - ' output_folder '/SANDI-fit_' names{i} '.nii.gz'])

#     end

#     if diagnostics==1
#         r.close();
#     end
    
# catch

#     if diagnostics==1
#         r = SANDIinput.report{SANDIinput.subj_id};
#         h = figure('Name','SANDI maps for a representative slice'); hold on
#     end

#     for i=1:size(mpgMean,2)+3

#         itmp = zeros(sx*sy*sz,1);

#         if i==1
#             itmp(mask==1) = fneurite;
#         elseif i==2
#             itmp(mask==1) = fsoma;
#         elseif i==size(mpgMean,2)+1
#             itmp(mask==1) = fextra;
#         elseif i==size(mpgMean,2)+2
#             Rsoma_tmp = mpgMean(:,4);
#             Rsoma_tmp(fsoma<=0.15) = 0;
#             itmp(mask==1) = Rsoma_tmp;
#         elseif i==size(mpgMean,2)+3
#             Din_tmp = mpgMean(:,3);
#             Din_tmp(fneurite<0.10) = 0;
#             itmp(mask==1) = Din_tmp;
#         else
#             mpgMean(mpgMean(:,i)<0,i) = 0;
#             itmp(mask==1) = mpgMean(:,i);
#         end

#         itmp = reshape(itmp,[sx sy sz]);

#         if diagnostics==1
#             if i<=size(mpgMean,2)+1
#                 [~, ~, slices, ~] = size(itmp);
#                 slice_to_show = round(slices/2);
#                 slice_to_show(slice_to_show==0) = 1;
#                 subplot(2,3,i), hold on, imshow(itmp(:,:,slice_to_show), bounds(i,:)), title(names{i}), colorbar, colormap parula
#             end
#         end

#         nifti_struct.img = itmp;
#         nifti_struct.hdr.dime.dim(5) = size(nifti_struct.img,4);
#         if size(nifti_struct.img,4)==1
#             nifti_struct.hdr.dime.dim(1) = 3;
#         else
#             nifti_struct.hdr.dime.dim(1) = 4;
#         end
#         nifti_struct.hdr.dime.datatype = 16;
#         nifti_struct.hdr.dime.bitpix = 32;

#         save_untouch_nii(nifti_struct,fullfile(output_folder, ['SANDI-fit_' names{i} '.nii.gz']));
#         disp(['  - ' output_folder '/SANDI-fit_' names{i} '.nii.gz'])

#     end

#     if diagnostics==1

#         T = getframe(h);
#         imwrite(T.cdata, fullfile(output_folder , 'SANDIreport', 'SANDI_Maps.tiff'))

#         r.section('SANDI maps for a representative slice');
#         r.add_text('The plot shows the SANDI parametetric maps for a representative slice.');
#         r.add_figure(gcf,'SANDI parametetric maps for a representative slice','left');
#         r.end_section();
#         r.close();

#         savefig(h, fullfile(output_folder, 'SANDIreport', 'SANDI_Maps.fig'));
#         close(h);
#     end

# end

# end