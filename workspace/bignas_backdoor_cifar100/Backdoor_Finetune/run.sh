#!/bin/bash
T=`date +%m%d%H%M`

ROOT=~/run/SuperAdvNet/prototype/
export PYTHONPATH=$ROOT:$PYTHONPATH

#PARTITION=$1
NUM_GPU=$1
CFG=config2
if [ -z $2 ];then
    NAME=default
else
    NAME=$2
fi

module load anaconda
source activate nlp
export PYTHONUNBUFFERED=1
python3 -u -m prototype.solver.bignas_cifar10_backdoor_solver --config $CFG.yaml --phase finetune_subnet_backdoor \
#  2>&1 | tee log.train.$NAME.$T
