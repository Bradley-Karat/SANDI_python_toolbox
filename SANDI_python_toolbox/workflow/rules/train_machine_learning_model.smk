import subprocess
import numpy as np
import os

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir"
modeldir=f"{out_dir}/model"           

if not os.path.isdir(modeldir):           
    subprocess.call(['mkdir', modeldir])

path = expand(inputs['dwi'].input_path,zip,**inputs['dwi'].input_zip_lists)
pathbval = re.sub(".nii.gz", ".bval",path[0])
pathbvec = re.sub(".nii.gz", ".bvec",path[0])

pathcollect = f'{tmpdir}/collect_done.txt'

rule setup_and_train_model:
    input:
        bval = pathbval,       
        bvec = pathbvec,  
        connect = pathcollect     
    params:
        FWHM = str(config["FWHM"]),
        diravg = config["no_direction_averaging"],
        QC = config["use_QC"],
        Delta = config["Delta"],
        smalldelta = config["Small_Delta"],
        tmpdir = tmpdir,
        Dsoma = config["Dsoma"],
        Rsoma_UB = config["use_Rsoma_UB"],
        De_UB = config["De_UB"],
        Din_UB = config["Din_UB"], 
        MLmodel = config["ML_model"],
        FittingMethod = config["Fitting_method"],
        Nset = config["Nset_size"],
        method = config["MLP_predict_all"]
    log:
        log = os.path.join(out_dir,"model","train_machine_learning_model.log"),
    output:
        model=os.path.join(out_dir,"model","trained_model.pkl"),
        modelinfo=os.path.join(out_dir,"model","model_information.pkl"),
        training_set=os.path.join(out_dir,"model","database_training_set.npy"),
        QCnoise=os.path.join(out_dir,"model","Distribution_of_noise_variances.png"),          
    group:
        "subj"
    script:
        "../scripts/setup_and_run_model_training.py"