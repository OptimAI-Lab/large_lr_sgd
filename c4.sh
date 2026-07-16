# Download data

pip install git-lfs 

git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4

cd c4

git lfs pull --include "en/*" 
