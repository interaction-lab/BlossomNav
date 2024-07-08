# BlossomNav
## Requirements
- NodeJS and NPM Installed
- MiniConda Installed
- Python Installed

## Installation and Configuration
Clone the repository and its submodules (ZoeDepth):
```
git clone --recurse-submodules https://github.com/interaction-lab/BlossomNav.git
```
Install the dependencies from conda:
```
conda env create -n blossomnav --file environment.yml
conda activate mononav
```
**Tested On:** (release / driver / GPU)
<br />Ubuntu 22.04 / NVIDIA 535 / RTX 3060

## Running the Code
### Upload MP4:
You can run the following code if you want to use BlossomNav on
<br /> a video (.mp4) on your local computer.
```cd utils```
```python split.py video_file_path image_dir 10```
<br />The **video_file_path** is the path to your video and the
<br /> **image_dir** is the directory in which you want the images
<br /> to be saved. 
