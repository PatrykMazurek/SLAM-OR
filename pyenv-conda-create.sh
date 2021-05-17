#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME='pyslam'
fi

#conda create --name $ENV_NAME --file requirements-conda.txt -c conda-forge
#conda create --name $ENV_NAME --file list_env.txt -c conda-forge -c pip 
# or (easier)
conda env create -f pyslam_env.yml
#conda env create -f requirements-conda.yml
#conda env create -f packagelist.yml
#conda env create -n pyslam2 python=3.8 

# activate created env 
. pyenv-conda-activate.sh 

which pip  # this should refer to */pyslam/bin/pip  (that is actually pip3)
pip install -r requirements-conda-pip.txt  
