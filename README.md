# BlossomNav: A Software Suite for Mobile Socially Assistive Robots
**Authors**: [Anthony Song](https://anthonybsong.github.io/) and [Nathan Dennler](https://ndennler.github.io/)

_[Interaction Lab](https://uscinteractionlab.web.app/), University of Southern California_

[Project Page]() | [Poster](./_README/BlossomNav_Poster.pdf) | [Video] (https://youtu.be/rhi56NDJ_fc) |

---
Socially assistive robotics (SAR) aims to provide emotional, cognitive, and social support through robotic interactions. Despite the potential benefits, research and development in highly mobile SAR are limited, and existing solutions are often expensive and complex. BlossomNav aim to create hardware setup and software suite for SAR that is more affordable and user-friendly. It also aims to provide a base for future development of Visual SLAM software in the field of SAR by providing localization and mapping tools for robots that have built in cameras but do not have inertial measurement units (common in the field).

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
BlossomNav uses Intel's ZoeDepth model to estimate depth information from images. You can find the ZoeDepth model, scripts, etc under the ```ZoeDepth``` folder. To estimate depth, run: 
```
python estimate_depth.py
```
This script reads in images from the directory specified at ```data_dir``` in ```config.yaml``` and transforms them to match the camera intrinsics used in the ZoeDepth training dataset. These transformed images are saved in a directory called ```<camera_source>-rgb-images``` which are then used to estimate depth. Estimated depths are saved as numpy arrays and colormaps in ```<camera_source>-depth-image```.
Original Image                                          |  Transformed Image                                                 |  Depth Colormap
:------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------:
![Alt text](./_README/original.jpg?raw=true "Original") |  ![Alt text](./_README/transformed.jpg?raw=true "Transformed")     |  ![Alt text](./_README/depthcolormap.jpg?raw=true "Depth")

### Pose Estimation
BlossomNav attempts to estimate position using visual odometry. Most of the visual odometry functions can be found and changed at ```utils/utils.py``` if needed. To estimate a robots position, run 
```
python estimate_depth.py
```
This script uses the images stored at the path specified by ```data_dir``` in ```config.yaml``` to estimate a non-scaled rotation and translation matrix between two images. Then the script uses the depth arrays calculated stored at ```<camera_source>-depth-images``` to scale the matrices to real world metrics. Finally it estimates ground truth positions from the relative positions and transforms them into a 4 x 4 matrix that Open3D can use to create maps. The positions are stored as txts at ```<camera_source>-depth-images```. **To make sure the pose estimation works, the Open3D poses .txt should all look something like this (below)**:
```
9.997392317919935323e-01 -4.179400429904072192e-03 -2.244996721601764944e-02 -4.362497266418206149e-02
4.391199003202964080e-03 9.999462406613749410e-01 9.393250688446225932e-03 -8.801827270860801411e-02
2.240950216466347511e-02 -9.489383500959187867e-03 9.997038390510983863e-01 1.948284960918448938e+00
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
```

### Map Creation / Localization
To illustrate the fusion process, run the script:
```
python fuse_depth.py
```
The program processes the transformed images from depth estimation, the poses from visual odometry, and depths from the ZoeDepth model, and integrates them using Open3D's TSDF Fusion. Once finished, it will display a reconstruction with coordinate frames indicating the camera poses during the operation. The reconstruction will be saved as a VoxelBlockGrid file ```vbg.npz``` and a point cloud file ```pointcloud.ply```, which can be viewed in MeshLab.

The stable branch comes with images that you can run these three python scripts on. If you use these tutorial images, then you should get the following files added to your ```data```:
```
├── <demo_hallway>
│   ├── <pixel_rgb_images> # images transformed to match kinect intrinsics
│   ├── <pixel_depth_images> # estimated depth (.npy for fusion and .jpg for visualization)
│   ├── vbg.npz / pointcloud.ply # reconstructions generated by fuse_depth.py
```

You should also get the following result from MeshLab:
<br />![mapping](https://github.com/user-attachments/assets/96527946-96d1-4f44-9b95-103ee26a266e)


## Camera Calibration
In order to have correct metric depth estimation accuracy in the ZoeDepth model, we must transform the input image to match the intrinsics of the training dataset. ZoeDepth is trained on [NYU-Depth-v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), which used the Microsoft [Kinect](https://en.wikipedia.org/wiki/Kinect). 

If you need to find where this is being done, ```utils/utils.py``` has the `transform_image()` which performs resizing the image and undistorting it to match the Kinect's intrinsics.

We provide scripts to generate `intrinsics.json` for your own camera. Steps to calibrate:

1. Take a video of a chessboard using ```app.py```. An example video can be found [here](https://www.youtube.com/watch?v=7N9aFjwYUy0&ab_channel=AnthonySong).
3. Use ```split.py``` to split the video into frames.
4. Use `calibrate.py`: Based on the [OpenCV sample](https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py).You need to provide several arguments, including the structure and dimensions of your chessboard target. Example:
    ```
    MonoNav/utils/calibration$ python calibrate.py -w 6 -h 8 -t chessboard --square_size=35 ./calibration_pictures/frame*.jpg

    ```
    The intrinsics are printed and saved to `utils/calibration/intrinsics.json`.
5. `transform.py`: Adapted from [MonoNav](https://github.com/natesimon/MonoNav). This script loads the intrinsics from `intrinsics.json` and transforms your `calibration_pictures` to the Kinect's dimensions (640x480) and intrinsics. This operation may involve resizing your image. The transformed images are saved in `utils/calibration/transform_output` and should be inspected.

## Acknowledgements
This work is heavily inspired by following works: 
<br />**Intelligent Robot Motion Lab, Princeton Unviersity** - [MonoNav](https://github.com/natesimon/MonoNav)
<br />**felixchenfy** - [Monocular-Visual-Odometry](https://github.com/felixchenfy/Monocular-Visual-Odometry)
<br />**isl-org** - [ZoeDepth](https://github.com/isl-org/ZoeDepth)
<br />**hrc2** - [blossom-public](https://github.com/hrc2/blossom-public)
