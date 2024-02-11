# Transparent Object Pose Refinement 
#### Master's thesis project 
This repo contains the implementation of my master's thesis project: Transparent Object Pose Refinement Pipeline using Differentiable Rendering.

[Thesis Link](https://drive.google.com/file/d/12WIZiDOofjRkkTLOMlQK83cBeU6k1cLg/view?usp=share_link)

![Transparent Object Pose Refinement Pipeline](https://github.com/negarl/Transparent-Object-Pose-Refinement/blob/master/fig/%E2%80%8Epipeline_V5.jpeg)


## Files 
- `main.py`: Runs the optimization on all images of the dataset. Outputs estimated object poses.
- `main_bop.py`: Runs the optimization on all images of the selected bop dataset. Outputs estimated object poses.
### Packages:
- `pose/`: The inverse rendering model and optimization loop.
- `mask/`: Everything related to the mask-based loss.
- `collision/`: Everything related to the collision-based loss.
- `contour/`: Everything related to the contour-based loss.
- `utility/`: Helper classes and functions.


## Dataset 

Details of the dataset can be found at [Dataset_README](./dataset/README.md)

## Installation
- `conda create -n pytorch3d python=3.9`
- `conda activate pytorch3d`
- `conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia`
- `conda install -c fvcore -c iopath -c conda-forge fvcore iopath`
- `pip install -r requirements.txt `
#### PyTorch3D
- `conda install pytorch3d -c pytorch3d`

For more up to date dependency information check the 
[Pytorch3d Website](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
