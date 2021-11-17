#!/bin/bash
T=`date +%m%d%H%M`

ROOT=~/run/SuperAdvNet/prototype/
export PYTHONPATH=$ROOT:$PYTHONPATH

#PARTITION=$1
NUM_GPU=$1
CFG=./config.yaml
if [ -z $2 ];then
    NAME=default
else
    NAME=$2
fi

rm -rf log* events/ checkpoints/ bignas results/
module load anaconda
source activate nlp
export PYTHONUNBUFFERED=1
python -u -m prototype.solver.bignas_cifar10_backdoor_solver \
  --config=$CFG \
  --phase train_supnet_backdoor\
  2>&1 | tee log.train.$NAME.$T
