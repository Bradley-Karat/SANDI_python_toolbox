# SANDI python toolbox
A BIDS app for fitting the Soma and Neurite Density Imaging (SANDI) model, adapted from the [SANDI Matlab Toolbox](https://github.com/palombom/SANDI-Matlab-Toolbox-Latest-Release/tree/main) by [Marco Palombo](https://github.com/palombom) into python using [snakebids](https://github.com/akhanf/snakebids).

This toolbox enables model-based estimation of MR signal fraction of brain cell bodies (of all cell types, from neurons to glia, namely soma) and cell projections (of all cell types, from dentrites and myelinated axons to glial processes, namely neurties ) as well as apparent MR cell body radius and intraneurite and extracellular apparent diffusivities from a suitable diffusion-weighted MRI acquisition using Machine Learning (see the original SANDI paper for more details DOI: https://doi.org/10.1016/j.neuroimage.2020.116835).
## Installation
Dependency management and python package handling is done with [poetry](https://python-poetry.org/docs/)
```
pip install poetry
```
then the toolbox can be installed with:
```
git clone https://github.com/Bradley-Karat/SANDI_python_toolbox.git
cd SANDI_python_toolbox
poetry install
```
## Usage
```
poetry run SANDI_python_toolbox
```
or 
```
poetry shell
SANDI_python_toolbox
```
Run either of these with the -h flag to get a detailed summary of the toolbox and its flags plus required arguments.
## Example Toolbox Call
For a dry-run:
```
poetry run SANDI_python_toolbox /path/to/bids/inputs /path/for/outputs participant --Delta 23.6 --Small_Delta 7 --cores all -np
```
To actually run the software:
```
poetry run SANDI_python_toolbox /path/to/bids/inputs /path/for/outputs participant --Delta 23.6 --Small_Delta 7 --cores all
```
## Example File Struture
```
└── bids
    ├── sub-01
    │ └── dwi
    │     ├── sub-01_brain_mask.nii.gz
    │     ├── sub-01_dwi.bval
    │     ├── sub-01_dwi.bvec
    │     ├── sub-01_dwi.nii.gz
    │     └── sub-01_noisemap.nii.gz
    ├── sub-02
    │ └── dwi
    │     ├── sub-02_brain_mask.nii.gz
    │     ├── sub-02_dwi.bval
    │     ├── sub-02_dwi.bvec
    │     ├── sub-02_dwi.nii.gz
    │     └── sub-02_noisemap.nii.gz
```
Alternatively, the `--path-dwi` flag can be used to override BIDS by specifying absolute paths. For example: `--path-dwi /path/to/my_data/{subject}/dwi.nii.gz`. The other necessary files (.bval, .bvec, noisemap and brainmask) must be labelled in a similar fashion (i.e. all have the same prefix).
