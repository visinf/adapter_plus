# Adapters Strike Back

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository of our [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html):

**Adapters Strike Back**<br>
[Jan-Martin O. Steitz](https://jmsteitz.de)
and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<br>
*CVPR*, 2024

**Abstract:** Adapters provide an efficient and lightweight mechanism for adapting trained transformer models to a variety of different tasks. However, they have often been found to be outperformed by other adaptation mechanisms, including low-rank adaptation. In this paper, we provide an in-depth study of adapters, their internal structure, as well as various implementation choices. We uncover pitfalls for using adapters and suggest a concrete, improved adapter architecture, called *Adapter+*, that not only outperforms previous adapter implementations but surpasses a number of other, more complex adaptation mechanisms in several challenging settings. Despite this, our suggested adapter is highly robust and, unlike previous work, requires little to no manual intervention when addressing a novel scenario. Adapter+ reaches state-of-the-art average accuracy on the VTAB benchmark, even without a per-task hyperparameter optimization.

<img src="https://github.com/visinf/adapter_plus/raw/main/assets/params_accuracy.png" width="512" />

## Training and evaluation

### Install requirements
```
conda env create -f environment.yml
conda activate adapter_plus
```

### Dataset preparation

For dataset preparation of the VTAB and FGVC benchmarks, please follow [VPT](https://github.com/KMnP/vpt). Our configuration expects the VTAB and FGVC dataset folders to reside under `datasets/vtab` and `datasets/fgvc`, respectively.

### Training

For training, you can select one of the preconfigured experiments from `conf/experiments` to train on the complete VTAB or FGVC benchmarks or define your own and run for example

```
python train.py +experiment=vtab/adapter_plus_dim1-32
```

### Results aggregation

To aggregate the results of the VTAB and FGVC benchmarks, you can use the Jupyter notebook `get_results.ipynb`.

### Evaluation of checkpoints

To evaluate checkpoints from previous training runs, use `eval.py` in combination with the experiment configuration you want to re-evaluate, e.g.:

```
python eval.py +experiment=vtab/adapter_plus_dim1-32
```

We also released [checkpoints](https://github.com/visinf/adapter_plus/releases/tag/v0.1.0) for our main experiments. To use them, create an `output` directory inside the repository's directory and extract the archives there. This should create a structure like `output/vtab/adapter_plus_dim8`. Then evaluate as described above.

### Disclaimer

Training and evaluation for the paper were done with timm 0.6.7, pytorch 1.12, and python 3.9. For the best possible usability of our pip module, we have updated the code to the latest versions. As such, the numbers on VTAB may vary slightly (we measured up to +/- 0.2 p.p. in accuracy). However, the global average accuracy across all VTAB subgroups remains unchanged.

## Use Adapter+ in your own project

To use our adapter implementation in your own project simply, install the pip module:

```
pip install adapter-plus
```

Besides various adapter configurations, our module also supports LoRA (without matrices merging for inference) and VPT-deep. Please refer to the configurations in the repository's `conf` directory for details.
Our pip module patches the `_create_vision_transformer` function of the timm library to support adapters. All vision transformers built with `Block` or `ResPostBlock` block functions are supported. 

You can create an adapter-enabled vision transformer model as shown below:

```python
import timm
import adapter_plus
from omegaconf import OmegaConf

# create config for Adapter+
# change bottleneck dim as required
adapterplus_conf = OmegaConf.create(
    """
    config: post
    act_layer: true
    norm_layer: false
    bias: true
    init: houlsby
    scaling: channel
    dim: 8
    attn_adapter: false
    dropout: 0
    drop_path: 0.1
    """
)

# create pre-trained timm ViT model 
# with adapter=True and adapter_config
model = timm.create_model(
    "vit_base_patch16_224.orig_in21k",
    adapter=True,
    pretrained=True,
    drop_path_rate=0.1,
    num_classes=101,
    adapter_config=adapterplus_conf,
)

# only require gradients for
# adapters and classifier head
model.requires_grad_(False)
model.head.requires_grad_(True)
for m in model.modules():
    if isinstance(m, adapter_plus.Adapter):
        m.requires_grad_(True)
```

## Acknowledgements
This work has been funded by the LOEWE initiative (Hesse, Germany) within the [emergenCITY](https://www.emergencity.de/) center.

## Citation
```
@inproceedings{Steitz:2024:ASB,
  author    = {Steitz, Jan-Martin O. and Roth, Stefan},
  title     = {Adapters Strike Back},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {23449--23459}
}
```