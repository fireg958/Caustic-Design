# Caustic-Design-Project

There are two major parts in this project. 

 1.  Caustic-Design - handles Optimal-Transport and corresponding tasks
 2.  Target Optimization - handles 3D Optimization


## Caustic-Design

Note: This step is optional.. qmake is not fully functional in the current version. Please see the Build-Section for details.

<b>Import in QT-Creator</b>
 *  Open QT-Creator
 *  Choose File > Open File or Project 
 *  Navigate to Caustic_Design folder
 *  Choose Caustic_Design.pro and confirm Dialog with 'Open'
<br>

## Dependencies

Following dependencies are needed for the Caustic_Designer:<br>

 *  cmake
 *  libqt4-dev (needs to be tested)
 *  libblas-dev
 *  liblapack-dev
 *  libtbb-dev
 *  libmetis-dev
 *  libsuitesparse-dev (or manually via: http://faculty.cse.tamu.edu/davis/suitesparse.html)
 *  liblbfgs-dev
 *  libtinyxml-dev

###Debian dependencies:<br>
`sudo add-apt-repository ppa:rock-core/qt4` <br>
`sudo apt-get install cmake libqt4-dev libblas-dev liblapack-dev libtbb-dev libmetis-dev build-essential libsuitesparse-dev liblbfgs-dev libtinyxml-dev libgmp3-dev libmpc-dev libboost-all-dev checkinstall`

install CGAL 4.9.1:
`mkdir ~/source` <br>
`cd ~/source` <br>
`wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.9.1/CGAL-4.9.1.tar.xz` <br>
`tar -xf CGAL-4.9.1.tar.xz` <br>
`cd build-CGAL-4.9.1` <br>
`cmake ../CGAL-4.9.1` <br>
`make` <br>
`sudo checkinstall` <br>


## Build
We suggest using cmake to build the project. To do so, simple:

 1.  Create Build Directory (e.g. `mkdir build-Caustic_Design`)
 2.  Run cmake in the build directory (e.g. `cd build-Caustic_Design && cmake ../Caustic_Design/`)
 
Check for the existence of the dynamic library path environment variable(LD_LIBRARY_PATH)
`echo $LD_LIBRARY_PATH`

If there is nothing to be displayed, add a default path value (or not if you wish to)
`LD_LIBRARY_PATH=/usr/local/lib`


You may also use qmake instead of cmake if you prefer qmake. But in the current version, suitesparse does not seem to be set correctly when installing it via `apt-get`.


## Usage

Following sections describe the usage of the different steps for the caustic-design project.

### Optimal Transport

To run the optimal transport, a source-image as well as a target-image needed. The rest of the steps are done automatically.<br>

----------

<b>Loading source- and target-image</b><br>
 *  Via code:  In window.cpp uncomment the lines `//open(QString("/home/p/Pictures/einstein_small.png"), false);` and `//open(QString("/home/p/Pictures/white_small.png"), true);` and replace the strings with the path to the target image and source image, respectively.
 *  Via UI: `File > Load Image` to load target image and `File > Load Source Image` to load source image.

<b>NOTE</b>: Source- and Target-Image need to be of same ratio.

---------

<b>Configuration</b><br>
All relevant values (amount of sites, multi-scale levels) can be configured in the config.h file.

---------

<b>Running</b>
To finally run the Optimal Transport: 
 *  `Algorithm > Compute Optimal Transport`
 *  When running Gradient Descent, ensure that `LEVEL_MAX` is set to `1`


------

### Interpolation-> Natural Neigbors CGAL 
// Assigned to: Cam<br>
algorithm->compute interpolation to load the source image, and points (.dat) and weights (.weights) data from the OTM and run the interpolation

------

## Target Optimization

## Dependencies
Following libraries are needed:

 *  ceres-solver: http://ceres-solver.org/installation.html
 *  libassimp-dev

### Target Surface -> C++ (3D part) 
// Assigned to: Cam<br>
<b>Input</b>
 *  Coordinates (xR) [4]
 *  Target Light Direction (calculated from [4])
 *  Surface Mesh
 *  Incoming Light Direction

<b>Output</b>
 *  3D Mesh (target surface) 

<b>Computing the surface optomization</b><br />
The code is located in target-surface-optimization folder<br>
Prerequisits:<br>
 *  openGL
 *  GLM header librairy
 *  glew
 *  glfw
 * SOIL
 * assimp

Debian dependencies as one-liner:<br>
`sudo apt-get install libglew-dev libsoil-dev`
<br>

## HOW TO USE
### Compute Optimal Transport
- First create your two grayscale images. The first image is the target image. This is the image that you want your lens to project. The second image is the source image. That is what the projection currently looks like which is in most cases a blank white image.
- Open Caustic_Design and click `File -> Load Image`, then load your target image. Next, click `File -> Load Source Image`, then load your source image.
- To set optimal transport resolution, click `Algorithm -> Set Parameters` and change the `Sites` from 5000 to any other value. The higher this number is, the better your results will be, but the longer the optimal transport and the other steps will take. You should also increase the `Levels`.
- Now finally click on `Algorithm -> Compute Optimal Transport`, click yes, and now the software will calculate the optimal transport between the source image and target image. It will take a couple of hours to complete depending on your specified steps and resolution. When it finishes computing, it will ask you to store the .weights file, save this file. You should also save the .dat file which you can do by clicking `File -> Save Source DAT`.

### Compute Interpolation
- Once you computed the optimal transport, the next thing to do is create a mesh in blender that will be deformed to create your caustic surface. I got the best results creating a cube scaling it on the x-axis (0.4x1.0x1.0 for example), open it in edit mode, select the face facing the positive x-axis, click edge->subdivide, enter the amount of subdivisions you want, more is better, but is slower to compute, then triangulate all faces, and export it as .obj.
- Next, launch the Target_Surface program and click `File -> Load Model`, then load your exported 3d model from belnder, check to see if the subdivided face is facing the blue axis line, and click `File -> Save Vertices`, save this file for the next step.
- Next, go back to Caustic_Design (this can be a new instance but doesnt have to be), Click on `Algorithm -> Compute Interpolation`, it will ask you for the target image again (the image you want to project), pointset (perviously generated .dat), weights (previously generated .weights), and light origin points (vertices generated from the 3d mesh).
- when the last file is loaded, it will start the interpolation process. This will take a long time depending on the resolution of the mesh and the amount of sites from the optimal transport step.
- When it is finished, it will ask you to save the interpolated point set (.dat), this file will be used to generate the caustic surface in the next step.

## Generate the Caustic Surface
- Open a new instance of the Target_Surface program and load your model again, then click on `File -> Load Light-Ray Receiver Position` open the .dat file from the interpolation step, if you then rotate the view, you should see your caustic image apear on the plane. if this is the case, change your desired focal length and click `Algorithm -> Run Target Optimization`. This will modify the surface of the model so it will project the caustic image. This will take a while depending on how easy it is to solve the surface hightmap.
- When the surface solver is done, you can export the model by clicking `File -> Save Model`, save the file as .stl
- ?
- Profit