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
    - A multi-view video file (.mp4) with 8 views
    - An attention file in .txt format (starts with "attC_") and contain the DevEv session and subject number in the name: "##_##". The attention file contains head andhands location as well as attention.
    - A toy file (.npy) containing the coordinate in 3D of the centroid of toys for each frame
### There are 3 correction tools for manipulating the data
    - The attention and head location correction
    - The left and right hands location correction
    - The toys centroid location correction
### For a triplet of video, attention and toy files
    - The interface display in real time the 3D representation of the scene including the infant data as well as moving toys
    - The interface can project the 3D data into 2D and display the results on the video using the camera parameters by using the "Compute2D" menu button
### A visualization tool is available for displaying multiple attention file at once
    - By clicking on the "Visualizer" menu button a window will appear allowing to select a directory containing multiple attention files (head, attention, left hand, right hand)
    - The files will be sorted into categories (room self play, mat self play, parent play) based on the session and subject number. If the file does not start with "attC_" and does not contained the session and subject number in the name as "##_##", it will be filtered out.
    - It is possible to select what type of data to display (head, attention vector/cone, left hand, right hand)
    - Everytime a new selection of file is done, the user should click the "Display" button to update the visualization.


