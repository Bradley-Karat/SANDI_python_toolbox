import subprocess
import numpy as np
import os

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir"

if not os.path.isdir(tmpdir):           
    subprocess.call(['mkdir', tmpdir])
    np.save(f'{tmpdir}/hold_noisemap_norm_mppca.npy',np.empty((0)))
    np.save(f'{tmpdir}/hold_noisemap_norm_SHresiduals.npy',np.empty((0)))

def get_script(use_SNR):
    if use_SNR:
        script_path =  "../scripts/make_direction_average_SNR.py"
    else:
        script_path = "../scripts/make_direction_average_noisemap.py"
    return script_path

script_path = get_script(config["use_SNR"])

rule make_direction_average:
    input:
        dwi = inputs.input_path["dwi"],
        mask = re.sub("dwi.nii.gz", "brain_mask.nii.gz",inputs.input_path["dwi"]),       
        bval = re.sub(".nii.gz", ".bval",inputs.input_path["dwi"]),       
        bvec = re.sub(".nii.gz", ".bvec",inputs.input_path["dwi"]),       
        noisemap = re.sub("dwi.nii.gz", "noisemap.nii.gz",inputs.input_path["dwi"]),   
    params:
        FWHM = str(config["FWHM"]),
        diravg = config["no_direction_averaging"],
        QC = config["use_QC"],
        Delta = config["Delta"],
        smalldelta = config["Small_Delta"],
        tmpdir = tmpdir,
        use_script = get_script
    log:
        log = bids(root="logs", suffix="process_all_datasets.log", **inputs.input_wildcards["dwi"]),
    output:
        direction_avg=bids(
            root="work",
            suffix="diravg_signal.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),
        noisemap=bids(
            root="work",
            suffix="noisemap_from_SHfit.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),    
        QC = bids(
                root="QC",
                suffix="Spherical_Mean_Signal.png",
                **inputs.input_wildcards["dwi"]
            ),
        done = touch(temp(bids(
                root="work",
                datatype="dwi",
                suffix="done.txt",
                **inputs.input_wildcards["dwi"]
            ),
        ),
        )
    group:
        "subj"
    script:
        script_path

pathcollect = f'{tmpdir}/collect_done.txt'

rule collect_done:
    input:
        done = expand(rules.make_direction_average.output.done,zip,**inputs['dwi'].input_zip_lists),
    output:
        collect = pathcollect,
    shell:
        "cat {input.done} > {output.collect}"


