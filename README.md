# diffusion_pytorch

Experimentation with Diffusion based neural networks referencing various online resources such as:
- https://www.youtube.com/watch?v=a4Yfz2FxXiY&ab_channel=DeepFindr
- https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/
- and a lot of help from ChatGPT!

# Usage

## Set up environment

1. [Install Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Clone project & create conda environment:
```conda
git clone https://github.com/moddyz/diffusion_pytorch.git
cd diffusion_pytorch

# To create the environment for running on CPU
conda env create -f environment_cpu.yml
conda activate diffusion_pytorch_cpu

# For CUDA
conda env create -f environment_cuda.yml
conda activate diffusion_pytorch_cuda
```

# Experiments

1. [image_generator](./image_generator): Trains on a set of images then generates new images of trained characteristics.


