import os
import subprocess

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir" 

rule apply_model:
    input:
        model=os.path.join(out_dir,"model","trained_model.pkl"),
        modelinfo=os.path.join(out_dir,"model","model_information.pkl"),
        direction_avg=bids(
            root="work",
            suffix="diravg_signal.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),    
        mask = re.sub("dwi.nii.gz", "brain_mask.nii.gz",inputs.input_path["dwi"]),       
        bvals = re.sub(".nii.gz", ".bval",inputs.input_path["dwi"]),       
    params:
        QC = config["use_QC"],
        Delta = config["Delta"],
        smalldelta = config["Small_Delta"],
        MLdebias = config["no_ML_debias"],
        MLmodel = config["ML_model"],
        method = config["MLP_predict_all"],
        tmpdir = tmpdir
    log:
        log = bids(root="logs", suffix="run_model_fitting.log", **inputs.input_wildcards["dwi"]),
    output:
        maps=expand(
            bids(
                root="results",suffix="{metric}.nii.gz",desc="SANDI-fit",**inputs.input_wildcards["dwi"]
                ),
                metric=config['metrics'],allow_missing=True
            ),          
    group:
        "subj"
    script:
        "../scripts/run_model_fitting.py"