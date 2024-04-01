eval "$(/work/miniconda3/bin/conda shell.bash hook)"
conda init
export MY_ENV=myenv
echo "conda activate $MY_ENV" >> ~/.bashrc