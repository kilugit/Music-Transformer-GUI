## What's new?

I've optimized the source code to run on local hardwares, reducing VRAM consumption and increasing GPU acceleration.
I've also added a graphical interface and one-click installation scripts.
I've also revamped the folder management.
Simply put, it's now easier to use and install.

## Setting up
Clone, the git repository, cd into it if necessary, and install the requirements. Then you're ready to preprocess MIDI files, as well as train and generate music with a Music Transformer.
```shell
git clone https://github.com/kilugit/Music-Transformer-GUI
cd ./MIDI-Transformer-GUI
```

## Manual Installation:
Create env:
```
python3 -3.12 -m venv venv
venv/Scripts/activate
```

For AMD GPUs (ROCm):
```
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
	
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
	
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
