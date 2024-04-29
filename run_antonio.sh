#!/bin/bash

PYTHONPATH=$PYTHONPATH:.

# **************************************************************************************
# Model Assessment *********************************************************************
# **************************************************************************************

##################################
# MOONS - NI
##################################

## MOONS - SLDA
## python model_assessment_setup.py "/raid/carta/cl_dpmm/MOONS/SLDA/ni/2024_03_20_17_21_49"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/SLDA/ni/FINAL_RUNS/2024_03_27_16_03_35/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## MOONS - CN-DPM
## python model_assessment_setup.py "/raid/carta/cl_dpmm/MOONS/CN-DPM/2024_04_03_10_57_01"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/CN-DPM/FINAL_RUNS/2024_04_04_09_43_15/${r}/config.yaml"
#  python trainers/cn_dpm_trainer.py --config-file $config_file
#done

### MOONS - VAE
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/VAE/FINAL_RUNS/2024_03_26_15_59_51/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done

##################################
# CIFAR100
##################################

## CIFAR100 - SLDA
## python model_assessment_setup.py "/raid/carta/cl_dpmm/CIFAR100/SLDA/2024_03_15_15_45_44"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/SLDA/FINAL_RUNS/2024_03_27_16_53_05/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## CIFAR100 - CD-IMM
# python model_assessment_setup.py /raid/carta/cl_dpmm/CIFAR100/CD-IMM/2024_04_11_10_53_21
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/CD-IMM/FINAL_RUNS/2024_04_15_14_52_14/${r}/config.yaml"
#  python trainers/cd_imm_trainer.py --config-file $config_file
#done

## MOONS - VAE
## python model_assessment_setup.py "/raid/carta/cl_dpmm/CIFAR100/VAE/2024_03_22_15_32_04"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/VAE/FINAL_RUNS/2024_03_27_10_12_12/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done

## CIFAR100 - CN-DPM
## python model_assessment_setup.py "/raid/carta/cl_dpmm/CIFAR100/CN-DPM/2024_03_27_00_48_21"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/CN-DPM/FINAL_RUNS/2024_04_04_09_51_54/${r}/config.yaml"
#  python trainers/cn_dpm_trainer.py --config-file $config_file
#done


##################################
# OMniglot
##################################

## OMNIGLOT - SLDA
## python model_assessment_setup.py "/raid/carta/cl_dpmm/OMNIGLOT/SLDA/2024_03_15_16_39_35"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/SLDA/FINAL_RUNS/2024_03_27_17_05_57/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## OMNIGLOT - CD-IMM
## python model_assessment_setup.py "/raid/carta/cl_dpmm/OMNIGLOT/CD-IMM/2024_04_12_08_35_37"
for r in {0..4}
do
  config_file="/raid/carta/cl_dpmm/OMNIGLOT/CD-IMM/FINAL_RUNS/2024_04_17_14_40_03/${r}/config.yaml"
  python trainers/cd_imm_trainer.py --config-file $config_file
done

## OMNIGLOT - VAE
## python model_assessment_setup.py "/raid/carta/cl_dpmm/OMNIGLOT/VAE/2024_03_27_12_18_57"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/VAE/FINAL_RUNS/2024_04_03_09_51_54/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done

## OMNIGLOT - CN-DPM
## python model_assessment_setup.py "/raid/carta/cl_dpmm/OMNIGLOT/CN-DPM/2024_03_26_19_42_45"
#for r in {0..4}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/CN-DPM/FINAL_RUNS/2024_04_04_09_54_43/${r}/config.yaml"
#  python trainers/cn_dpm_trainer.py --config-file $config_file
#done


# **************************************************************************************
# Grid Search **************************************************************************
# **************************************************************************************

##################################
# MOONS - NI
##################################
# MOONS - SLDA
#for r in {0..1}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/SLDA/ni/2024_03_20_18_34_55/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## MOONS - VAE
#for r in {0..17}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/VAE/2024_03_22_15_19_21/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done

# MOONS-NI - DPMM
#for r in {0..53}
#do
#  config_file="/raid/carta/cl_dpmm/MOONS/NI/dpmm/2024_03_20_18_34_55/${r}/config.yaml"
#  python trainers/cd_imm_trainer.py --config-file $config_file
#done

##################################
# CIFAR100
##################################
# CIFAR100 - SLDA
#for r in {0..1}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/SLDA/2024_03_15_15_45_44/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## CIFAR100 - VAE
#for r in {0..53}
#do
#  config_file="/raid/carta/cl_dpmm/CIFAR100/VAE/2024_03_22_15_32_04/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done

##################################
# Omniglot
##################################
# OmniGlot - SLDA
#for r in {0..1}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/SLDA/2024_03_15_16_39_35/${r}/config.yaml"
#  python trainers/slda_trainer.py --config-file $config_file
#done

## OMNIGLOT - DPMM
#for r in {0..53}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/DPMM/2024_03_20_18_36_01/${r}/config.yaml"
#  python trainers/cd_imm_trainer.py --config-file $config_file
#done

## OMNIGLOT - VAE
#for r in {0..53}
#do
#  config_file="/raid/carta/cl_dpmm/OMNIGLOT/VAE/2024_03_27_12_18_57/${r}/config.yaml"
#  python trainers/class_vae_trainer.py --config-file $config_file
#done