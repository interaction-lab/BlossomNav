# BlossomNav
## Software Requirements
- NodeJS and NPM Installed
- MiniConda Installed
- Python Installed

## Hardware Requirements
- Raspberry Pi Zero 2
- Raspberry Pi Camera Rev 1.3

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
You can run the following code if you want to use BlossomNav on a video (.mp4) on your local computer.
```
cd utils
python split.py video_file_path image_dir 10
cd .. // go back to the parent directory
```
The **video_file_path** is the path to your video and the **image_dir** is the directory in which
<br /> you want the images to be saved. 
<br />
### Using the Raspberry Pi Zero 2:
If you have set up the Raspberry Pi Zero 2, you can use our app to record videos to use in BlossomNav too.
```
python app.py
```
