# BERT-PRISM
Code will be updated later.

## **Setup environment**

```
# create and activate virtual python environment
conda create -n dna python=3.8
conda activate dna

# (optional if you would like to use flash attention)
# install triton from source
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build-time dependency
pip install -e .

# install required packages
python3 -m pip install -r requirements.txt
conda install -c conda-forge ncbi-datasets-cli

```

## **Run**

```

# single gpus
nohup python main.py --gpu_ids 0 1>/dev/null 2>&1 &

# multi gpus
nohup python main.py --gpu_ids 01 1>/dev/null 2>&1 &
```
