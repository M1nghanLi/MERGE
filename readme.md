**Multimodal reconstruction for compressive light field imaging via regularization by generation (MERGE)** is a groundbreaking self-supervised light field reconstruction framework. This repository includes the core source code, experimental data and silulation data for demonstrating and reproducing the results of MERGE

![KNIGHTS demo](images/KNIGHTS.gif)


## Project Scope

This project mainly covers the following task:

- Reconstruct 4D light fields from compressed measurements
- Improve reconstruction quality with multimodal priors


## System requirements
All our algorithms run under PyTorch 2.8 and utilize CUDA for acceleration, with the operating system being Ubuntu. To ensure the normal operation of all demonstration programs, we recommend using a GPU with at least 80GB of VRAM (the script already supports multi-GPU training)

## Installation guide
It is recommended to use Anaconda with Python 3.10 (as defined in `environment.yml`) and install dependencies with Conda.
```
conda env create -f environment.yml
conda activate MERGE
python demo_CSLIP_comparison.py
```

## Experiment data
https://drive.google.com/file/d/1ApFo04zt4A5dkjB-pWpQxVmSFOWdyLKV/view?usp=sharing

## Checkpoints of pretained model
- Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2
- DRUNet: https://github.com/cszn/DPIR/tree/master


## Repository Structure (Brief)

- `MERGE_model.py`:     Core model definitions
- `Forward_Model.py`:   Forward model implementation
- `CreateData.py`:      Data generation
- `DenoisingLIB.py`: Denoising modules

- `demo_CLIP.py` and `demo_CodedAperture.py`: Demonstrate the reconstruction process of two different experiment data——compact light field photography and coded aperture light field photography
- `demo_CSLIP_comparison.py`: Using the compressive light field photograph model and comparison of MERGE and PnP-FISTA


## Quick Start (Examples)

The following commands are common entry-point examples. Please adjust parameters for your experiment settings:

```bash
python demo_CLIP.py
python demo_CodedAperture.py
python demo_CSLIP_comparison.py
```

You should choose the correct GPU devices for reconstruction
The experimental data from the Coded Aperture experiment is relatively large. For ease of demonstration, the script uses a default setting of 4x downsampling before light field reconstruction, and the disparity-depth conversion data has been correspondingly matched.

## Outputs and Visualization

Typical outputs include:

- Reconstructed images/videos (e.g., png, gif, mp4)
- Disparity/depth maps
- Light-field tensors (`.pt`)
