# Material tokenization (MP-20)

FlowVQ and baseline tokenizers (VQ-VAE, LFQ, OptVQ) for crystal structures from MP-20.

## Requirements

The code was tested with CUDA 12.4. If you use a different CUDA version, install PyTorch and PyG
(lines 3-4 below) accordingly; see the official installation guides for
[PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
conda create --name flowvq python=3.10
conda activate flowvq
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -r requirements.txt
```

All commands below are run from this directory. The default configuration is `configs/mp_20.yml`;
any entry can be overridden with `--conf_new key=value ...`. Runs are logged to
`logs/mp_20/<run_name>/` and to Weights & Biases (`wandb.project` in the config).
On first use the CIF files in `data/mp_20/*.csv` are parsed and cached as
`data/mp_20/prepared_*_data_12.pt`, and the lattice scaler is saved to `data/mp_20/matrix_scaler.pt`.

Training uses a single GPU (select it with `CUDA_VISIBLE_DEVICES`). Use a distinct `--port` for
every run that executes concurrently on the same machine.

## FlowVQ tokenizer

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --conf_new model.flow=True model.optvq=False model.lfq=False model.flow_start_epoch=501
```

The model is trained as a plain auto-encoder for 500 epochs. At epoch `model.flow_start_epoch` (501) it
saves the auto-encoder as `saved_model/ae_500.pt`, initializes the codebook with the flow
(saving `saved_model/init_flow.pt` and `saved_model/optim_init_flow.pt`) and continues with vector
quantization, refreshing the codebook every `model.flow_update_every` epochs. The log directory is
`logs/mp_20/flow_K8192Z8E501F5e_scTrue_ocTrue`.

## Baseline tokenizers

The baselines start from the same auto-encoder `ae_500.pt` saved by the FlowVQ run above
(or from the pretrained one, see below). In the commands below

```bash
AE=logs/mp_20/flow_K8192Z8E501F5e_scTrue_ocTrue/saved_model/ae_500.pt
```

### 1. K-means codebook initialization

The VQ-VAE and OptVQ codebooks are initialized with k-means centroids of the auto-encoder latents:

```bash
# Encode the training set -> saved_model/ae_emb_500.pt
python3 get_latent_emb.py --ckpt $AE --conf_new model.flow=True model.flow_start_epoch=501
# K-means with K=8192 -> saved_model/ae_centroids_8192_500.pt
python3 get_kmeans.py --ckpt $AE
```

With `model.kmeans_init=True` (the default) the centroids are loaded from the file that matches
`model.ckpt` and `model.code_num`.

### 2. Baseline training

#### K-means initialized VQ-VAE

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --conf_new model.latent_dim=8 model.ckpt=$AE --port 11113
```

#### Auto-encoder initialized LFQ

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --conf_new model.flow=False model.optvq=False model.lfq=True model.ckpt=$AE model.code_weight=1 --port 11114
```

#### K-means initialized OptVQ

Change `model.optvq_eps` to explore the effect of epsilon in Eq. (8) of the OptVQ paper:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --conf_new model.flow=False model.optvq=True model.lfq=False model.ckpt=$AE model.kmeans_init=True model.optvq_eps=10 --port 11115
```

### Resume FlowVQ from a flow-initialized codebook

The flow-initialized codebook saved by the FlowVQ run can be loaded to start vector quantization directly,
e.g. for ablations (here without online correction):

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --conf_new model.flow=True model.flow_start_epoch=501 model.online_correction=False model.ckpt=$AE model.flow_init_ckpt=logs/mp_20/flow_K8192Z8E501F5e_scTrue_ocTrue/saved_model/optim_init_flow.pt --port 11116
```

## Pretrained auto-encoder

The auto-encoder pretrained for 500 epochs, together with its k-means and flow-trained codebook
initializations, can be downloaded from
[here](https://drive.google.com/drive/folders/1NNBiE3altMme6z8sMlbt_Rmxc5l-ewK1?usp=drive_link).
Place the files in `logs/mp_20/flow_K8192Z8E501F5e_scTrue_ocTrue/saved_model/`.

