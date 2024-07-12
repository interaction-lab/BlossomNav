# BlossomNav
## Requirements
### Software
- Python
- MiniConda / Anaconda
- NodeJS & NPM (If you want to use the JavaScript Image Downloader)

### Hardware
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
<br />Ubuntu 20.04 / NVIDIA 535 / RTX 2060

## Running the Code
### Using an MP4:
If you want to use BlossomNav's localization and mapping on a video (.mp4) on your local computer, you can run the following code.
```
cd utils
python split.py video_file_path image_dir 10
cd .. // go back to the parent directory
```
The **video_file_path** is the path to your video, and the **image_dir** is the directory in which you want the images to be saved. The last input of split is the occurence of saving an image. In this case the 10 means that a image is saved every 10 frames. We recommend playing around with this last parameter. 
<br />
### Using the Raspberry Pi Zero 2:
If you have set up the Raspberry Pi Zero 2, you can also use our one of our two methods to record videos to record from the pi. First go into the app.py file and choose whether you want a GUI by setting ```GUI = 0``` or ```GUI = 1``` in the file. **Note, we recommend turning GUI off to decrease latency in case you want to teleoperate the robot with the joystick**. Then you can run:
```
python app.py
```
Below is an image of the app's user interface. You can press Start to start recording and stop when you want to by pressing the stop button. After hitting stop, you will get the following pop-up screen. We recommend saving the mp4 into the **data** folder and following the steps from **Upload MP4**.
User Interface                                |  Save Screen
:--------------------------------------------:|:------------------------------------------------------------:
![Alt text](./_README/gui.png?raw=true "GUI") |  ![Alt text](./_README/savescreen.png?raw=true "Save Screen")

## Acknowledgements
<br />This work is heavily inspired by following works: 
<br />**Intelligent Robot Motion Lab, Princeton Unviersity** - [MonoNav](https://github.com/natesimon/MonoNav)
<br />**felixchenfy** - Monocular-Visual-Odometry - [Monocular-Visual-Odometry](https://github.com/felixchenfy/Monocular-Visual-Odometry)
