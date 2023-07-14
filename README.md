# SANDI python toolbox
A BIDS app for fitting the Soma and Neurite Density Imaging (SANDI) model, adapted from the [SANDI Matlab Toolbox](https://github.com/palombom/SANDI-Matlab-Toolbox-Latest-Release/tree/main) by [Marco Palombo](https://github.com/palombom) into python using [snakebids](https://github.com/akhanf/snakebids).

This toolbox enables model-based estimation of MR signal fraction of brain cell bodies (of all cell types, from neurons to glia, namely soma) and cell projections (of all cell types, from dentrites and myelinated axons to glial processes, namely neurties ) as well as apparent MR cell body radius and intraneurite and extracellular apparent diffusivities from a suitable diffusion-weighted MRI acquisition using Machine Learning (see the original SANDI paper for more details DOI: https://doi.org/10.1016/j.neuroimage.2020.116835).
## Installation
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
