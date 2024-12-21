#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J diffusion_model_image_generation
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need XGB of memory per core/slot --
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
##BSUB -M 20GB
### -- set walltime limit: hh:mm --
#BSUB -W 01:00
### -- set the email address --
#BSUB -u s224190@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o diffusion_model_image_generation_%J.out
#BSUB -e diffusion_model_image_generation_%J.err
# all  BSUB option comments should be above this line!

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# execute our command
source ~/.bashrc
conda activate diffusion_model
python /dtu/blackhole/12/186578/main.py