## Based on spectraldoy/music-transformer
A heavily modified version of Music Transformer with a graphical user interface (GUI) added.
https://github.com/spectraldoy/music-transformer

## What's new?

I've optimized the source code to run on local hardware, reducing VRAM consumption and increasing GPU acceleration.
I've added a graphical interface and one-click installation scripts.
I've also revamped the folder management.
Simply put, it's now easier to use and install.

## Setting up
Clone, the git repository, cd into it if necessary, and install the requirements. Then you're ready to preprocess MIDI files, as well as train and generate music with a Music Transformer.
```shell
git clone https://github.com/kilugit/Music-Transformer-GUI
cd Music-Transformer-GUI
```

## Manual Installation:
Create env:
```
py -3.12 -m venv venv
venv/Scripts/activate
```

For AMD GPUs (ROCm):
```
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-all]==10.0.0"
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.13.0+rocm10.0.0" "torchvision[device-gfx1201]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
pip install -r requirements.txt
```


For Intel Arc GPUs (XPU):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```

For AMD/Intel/NVIDIA GPUs (DirectML):
```
pip install torch-directml
pip install -r requirements.txt
```

## Launch GUI:
```
run.bat (Windows)
```
or
```
venv/Scripts/activate
py main.py
```
