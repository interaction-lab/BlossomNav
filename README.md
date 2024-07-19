# BlossomNav
## Requirements
### Software
- Python
- MiniConda / Anaconda
- NodeJS & NPM (Not necessary unless want to use JS in data collection)

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
conda activate blossomnav
```
Move to the stable branch:
```
git checkout stable
```
**Tested On:** (release / driver / GPU)
<br />Ubuntu 22.04 / NVIDIA 535 / RTX 3060
<br />Ubuntu 20.04 / NVIDIA 535 / RTX 2060

## BlossomNav: Data Collection Tools
### Using the Raspberry Pi Zero 2:
If you have set up the Raspberry Pi Zero 2 and Rev 1.3 Camera and other necessary hardware, you can use one of our three methods to download images from the pi camera - if not, please checkout the **hardware setup** section. Two of the methods use Python and one of them uses Javascript (JS). For the Python methods, one has a GUI and the other does not. **We recommend turning GUI off to decrease latency if you want to teleoperate the robot with our joystick**.
#### (1) Python Based - No GUI
First, go into the ```app.py``` file and set ```GUI = 0```.  After, run:
```
python app.py
```
Use ```Ctrl + L``` to start recording. A joystick will pop up allowing you to teleoperate a robot. Use ```Ctrl + C``` to exit the program and save the recording. The recording will be saved as output.mp4 to the directory specified at ```vid_dir``` in ```config.yaml```. The commands from the joystick will be saved as a txt to the directory specified at ```teleop_dir``` in ```config.yaml```.
#### (2) Python Based - GUI
First, go into the ```app.py``` file and set ```GUI = 1```.  After, run:
```
python app.py
```
Below is an image of the app's user interface. You can press Start to start recording and teleoperating the robot using our joystick. Pressing the stop button will stop the recording and terminate teleoperation. After hitting stop, you will be prompted on where to save the video recording. **We recommend saving the mp4 into the **data** folder. The commands from the joystick will be saved as a txt to the directory specified at ```teleop_dir``` in ```config.yaml```.
User Interface                                |  Save Screen
:--------------------------------------------:|:------------------------------------------------------------:
![Alt text](./_README/gui.png?raw=true "GUI") |  ![Alt text](./_README/savescreen.png?raw=true "Save Screen")
#### (3) JS Based
To use the downloadImage javascript file, run:
```
node downloadImage.js
```
This option does not yet have a joystick.
#### Joystick Information
[joystick.webm](https://github.com/user-attachments/assets/b111f9dd-f4b0-4e2f-aa10-ce3e5d1576a8)

Above is a video of our joystick. You can use your mouse to drag the inner circle within the outer circle. The joystick will send the x, y coordinates of the inner circle to the raspberry pi. The joystick will also snap back to (0, 0) if it does not sense that a mouse is dragging it. If you want to change the joystick to fit your system, the code can be found at ```datacollections/joystick.py```. The code for sending and receiving information from the joystick can be found in ```dataforwarding/datasender.py``` and ```dataforwarding/datareceiver.py```. You can alter these scripts to fit your system as well.

## BlossomNav: Video Parsing & Merging Tools
### Video Parsing
To use our video parsing tool, run:
```
cd utils
python split.py video_file_path image_dir 10
cd .. // go back to the parent directory
```
The **video_file_path** is the path to your video, and the **image_dir** is the directory in which you want the images to be saved. The last input of split is the occurence of saving an image. In this case the 10 means that a image is saved every 10 frames. **We recommend playing around with this last parameter**.
<br />
### Video Merging
To use our video merging tool, run:
```
cd utils
python merge.py image_frame_path output_dir 10
cd .. // go back to the parent directory
```
The **image_frame_path** is the path to your image frames, and the **output_dir** is the directory in which you want the merged .mp4 video to be saved. The last input of split is the frames per second. **We recommend playing around with this last parameter**.
<br />

## BlossomNav: Localization & Mapping Tools
Before you run BlossomNav's localization and mapping tools remember to calibrate your camera instrinsics. We have provided code to calibrate the intrinsics and directions can be found under the **Camera Calibration** section. **Camera calibration is crucial for depth estimation accuracy**. Also, if you have a specific .mp4 video you want BlossomNav to run on, you can save the video file to the data folder under ```data/recordings```**.

### Depth Estimation
BlossomNav uses Intel's ZoeDepth model to estimate depth information from images. You can find the ZoeDepth model, scripts, etc under the ```ZoeDepth``` folder. To estimate depth, run 
```
python estimate_depth.py
```
This script reads in images from the directory specified at ```data_dir``` in ```config.yaml``` and transforms them to match the camera intrinsics used in the ZoeDepth training dataset. These transformed images are saved in a directory called ```<camera_source>-rgb-images``` which are then used to estimate depth. Estimated depths are saved as numpy arrays and colormaps in ```<camera_source>-depth-image```.
Original Image                                          |  Transformed Image                                                 |  Depth Colormap
:------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------:
![Alt text](./_![Uploading original.jpg…]()             | ![Alt text](./_README/transformed.png?raw=true "Transformed") | ![Alt text](./_README/depthcolormap.png?raw=true "Depth")

### Pose Estimation
BlossomNav attempts to estimate position using visual odometry. Most of the visual odometry functions can be found and changed at ```utils/utils.py``` if needed. To estimate a robots position, run 
```
python estimate_depth.py
```
This script uses the images stored at the path specified by ```data_dir``` in ```config.yaml``` to estimate a non-scaled rotation and translation matrix between two images. Then the script uses the depth arrays calculated stored at ```<camera_source>-depth-images``` to scale the matrices to real world metrics. Finally it estimates ground truth positions from the relative positions and transforms them into a 4 x 4 matrix that Open3D can use to create maps. The positions are stored as txts at ```<camera_source>-depth-images```.

### Map Creation / Localization


## Camera Calibration

## Acknowledgements
This work is heavily inspired by following works: 
<br />**Intelligent Robot Motion Lab, Princeton Unviersity** - [MonoNav](https://github.com/natesimon/MonoNav)
<br />**felixchenfy** - Monocular-Visual-Odometry - [Monocular-Visual-Odometry](https://github.com/felixchenfy/Monocular-Visual-Odometry)
