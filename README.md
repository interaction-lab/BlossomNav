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
If you want to use BlossomNav on a video (.mp4) on your local computer, you can run the following code.
```
cd utils
python split.py video_file_path image_dir 10
cd .. // go back to the parent directory
```
The **video_file_path** is the path to your video, and the **image_dir** is the directory in which you want the images to be saved. 
<br />
### Using the Raspberry Pi Zero 2:
If you have set up the Raspberry Pi Zero 2, you can also use our app to record videos in BlossomNav.
```
python app.py
```
Below is an image of the app's user interface. You can press Start to start recording and stop when you want to by pressing the stop button.
<br />![Alt text](relative%20path/to/img.jpg?raw=true "Title")
Afterward, you can save the mp4 where you choose and follow the steps from **Upload MP4**
<br />![Alt text](relative%20path/to/img.jpg?raw=true "Title")
