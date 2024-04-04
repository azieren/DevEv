# DevEv

Visualization tool for watching in real time the 3D reconstruction of the visual attention of an infant locomoting in a room.
Visualizing, analizing and correcting 3D visual attention

## Installation
- Create a conda environment
```bash
conda create --name my_devev python=3.10
conda activate my_devev
```

- Install Dependencies
```bash
pip install -r requirement.txt
```

## Launching the interface

Open a python terminal by writing "python" in your command line or in a python script:

In the command interpreter:

```python
import DevEv
DevEv.run()
```

## Manual

### The interface can open 3 different types of files
- `File->Open Video`: Open multi-view video file (.mp4) with 8 views
- `File->Open Attention`: An attention file in .txt format (starts with "attC_") and contain the DevEv session and subject number in the name: "##_##". The attention file contains head andhands location as well as attention.
- `File->Open Toy file`: A toy file (.npy) containing the coordinate in 3D of the centroid of toys for each frame 
### For a triplet of video, attention and toy files
- The interface display in real time the 3D representation of the scene including the infant data as well as moving toys
- `File->Compute/Load 2D`: The interface can project the 3D data into 2D and display the results on the video using the camera 
### `View` menu can display up to 2 view in zoom in mode or all views
- `View->View 0`: Display the original video
- `View->View 1-8`: Display one or two of the views from the video
### There are 4 ways to display the room
- `Room View->Hide Room`: Remove the 3D room from the 3D visualization window
- `Room View->Wireframe Room`: Display the 3D room as a wireframe mesh
- `Room View->Transparent Room`: Display the 3D room as a transparent mesh
- `Room View->Solid Room`: Display the 3D room with solid colors and textures
### Setting the camera parameters for correction and `File->Compute/Load 2D`
- `Select Camera->Room View`: Load the camera parameters of the setup with the 8 cameras scattered in the room
- `Select Camera->Mat View`: Load the camera parameters of the setup with the 8 cameras focus on the checkerboard mat
### Setting the view in the 3D environment
- `Set 3D View->Cam 1-8`: Align the 3D room visualization with one of the view from the video
### There are 3 correction tools for manipulating the data
- `Correction->Attention/Head`: The attention and head location correction
- `Correction->Hands`: The left and right hands location correction 
- `Correction->Toys`: The toys centroid location correction 

### `Visualizer`: A visualization tool is available for displaying multiple attention file at once
- By clicking on button a window will appear allowing to select a directory containing multiple attention files (head, attention, left hand, right hand)
- The files will be sorted into categories (room self play, mat self play, parent play) based on the session and subject number. If the file does not start with "attC_" and does not contained the session and subject number in the name as "##_##", it will be filtered out.
- It is possible to select what type of data to display (head, attention vector/cone, left hand, right hand)
- Everytime a new selection of file is done, the user should click the "Display" button to update the visualization.


