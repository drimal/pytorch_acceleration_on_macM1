PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration. This MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. The MPS framework optimizes compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family. The new mps device maps machine learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.

# Set Up 

## Anaconda (on Mac M1)

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh

```
## Install torch with MPS support 
```conda install pytorch torchvision torchaudio -c pytorch-nightly```
or 
```pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu```

```
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```