

# Contrastive Unpaired Translation (CUT)

### [video](https://youtu.be/Llg0vE_MVgk) | [website](http://taesung.me/ContrastiveUnpairedTranslation/) |   [paper](https://arxiv.org/abs/2007.15651)
<br>

<img src='imgs/gif_cut.gif' align="right" width=960>

<br><br><br>



We provide our PyTorch implementation for unpaired image-to-image translation based on patchwise contrastive learning and adversarial learning.  No hand-crafted loss and inverse network is used. Compared to [CycleGAN](https://github.com/junyanz/CycleGAN), model training is faster and less memory-intensive. In addition, our method can be extended to single image training, where each “domain” is only a *single* image.




[Contrastive Learning for Unpaired Image-to-Image Translation](http://taesung.me/ContrastiveUnpairedTranslation/)  
 [Taesung Park](https://taesung.me/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/)<br>
UC Berkeley and Adobe Research<br>
 In ECCV 2020


<img src='imgs/patchnce.gif' align="right" width=960>

<br><br><br>

### Pseudo code
```python

import torch
cross_entropy_loss = torch.nn.CrossEntropyLoss()

# Input: f_q (BxCxS) and sampled features from H(G_enc(x))
# Input: f_k (BxCxS) are sampled features from H(G_enc(G(x))
# Input: tau is the temperature used in NCE loss.
# Output: PatchNCE loss
def PatchNCELoss(f_q, f_k, tau=0.07):
    # batch size, channel size, and number of sample locations
    B, C, S = f_q.shape

    # calculate v * v+: BxSx1
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S)[None, :, :]
    l_neg.masked_fill_(identity_matrix, -float('inf'))

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return NCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long)
    return cross_entropy_loss(predictions, targets)
```
## Example Results

### Unpaired Image-to-Image Translation
<img src="imgs/results.gif" width="800px"/>

### Single Image Unpaired Translation
<img src="imgs/singleimage.gif" width="800px"/>


### Russian Blue Cat to Grumpy Cat
<img src="imgs/grumpycat.jpg" width="800px"/>

### Parisian Street to Burano's painted houses
<img src="imgs/paris.jpg" width="800px"/>



## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting started

- Clone this repo:
```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CUT
```

- Install PyTorch 1.4 and other dependencies (e.g., torchvision, func-timeout, gputil).

For pip users, please type the command `pip install -r requirements.txt`.
For Conda users, we provide an installation script scripts/conda_deps.sh. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.


### CUT and FastCUT Training and Test

- Download the grumpifycat dataset (Fig 8 of the paper. Russian Blue -> Grumpy Cats)
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

The other datasets can be downloaded using
```bash
bash ./datasets/download_cut_dataset.sh [dataset_name]
```
, a script provided by the [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md) repo.

- Train the model:
```bash
# Trains the CUT model
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT

# Trains the FastCUT model
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_FastCUT --CUT_mode FastCUT
```
The checkpoints are stored at `./checkpoints/grumpycat_*/web`.

### Training using our launcher scripts

Please see `experiments/grumpifycat_launcher.py` that generates the above command line arguments.
The launcher scripts are useful for configuring rather complicated command-line arguments of
training and testing.

Using the launcher, the command below generates the training command of CUT and FastCUT.
```bash
python -m experiments grumpifycat train 0
python -m experiments grumpifycat train 1
```

To test using the laucher,
```bash
python -m experiments grumpifycat test 0
python -m experiments grumpifycat test 1
```

Possible commands are run, run_test, launch, close, and so on. Please see experiments/__main__.py for all commands



### Apply a pre-trained CUT model and evaluate

The tutorial for applying pretrained models will be released soon.

### SinCUT Single Image Unpaired Training

The tutorial for the Single-Image Translation will be released soon.



### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```


### Acknowledgments
We thank Allan Jabri and Phillip Isola for helpful discussion and feedback. Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation and [drn](https://github.com/fyu/drn) for mIoU computation, and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch/) for the PyTorch implementation of StyleGAN2 used in single-image translation.
