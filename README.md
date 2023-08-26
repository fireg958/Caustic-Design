# Caustic-Design-Project
Based on the original paper: [High-Contrast Computational Caustic Design](https://taiya.github.io/pubs/schwartzburg2014caustics.pdf)

There are two programs in this project.

 1.  Caustic-Design - handles Optimal-Transport and corresponding tasks
 2.  Target Optimization - handles 3D Optimization

## Goal
The goal of this project is to create an open source application that enables artists and researchers to compute the geometry of a surface such that its caustics cast a specified target distribution (an image for example).

![animation](./animation.gif)

## Building

Tested on Debian10 and Ubuntu 20.04. Because use of `ppa:rock-core/qt4`, it is not compatible with Ubuntu 22.04 and Debian11.
### Debian dependencies:<br>
`sudo apt update` <br>
`sudo add-apt-repository ppa:rock-core/qt4` <br>
`sudo apt install cmake libqt4-dev libblas-dev liblapack-dev libtbb-dev libmetis-dev build-essential libsuitesparse-dev liblbfgs-dev libtinyxml-dev libgmp3-dev libmpc-dev libboost-all-dev libglew-dev libsoil-dev libassimp-dev checkinstall`

install CGAL 4.9.1: <br>ls
`mkdir ~/source` <br>
`cd ~/source` <br>
`wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.9.1/CGAL-4.9.1.tar.xz` <br>
`tar -xf CGAL-4.9.1.tar.xz` <br>
`mkdir build-CGAL-4.9.1` <br>
`cd build-CGAL-4.9.1` <br>
`cmake ../CGAL-4.9.1` <br>
`make` <br>
`sudo checkinstall` <br>

install ceres-solver: http://ceres-solver.org/installation.html <br>
`cd ~/source` <br>
`wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz` <br>
`sudo apt update` <br>
`sudo apt-get install libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev` <br>
`tar zxf ceres-solver-2.1.0.tar.gz`  <br>
`mkdir ceres-bin`  <br>
`cd ceres-bin`  <br>
`cmake ../ceres-solver-2.1.0`  <br>
`make -j3`  <br>
`sudo make install`  <br>

### Build
We suggest using cmake to build the project. To do so, simple:

Caustic Design:<br>
`mkdir build-Caustic_Design` <br>
`cd build-Caustic_Design` <br>
`cmake ../Caustic_Design/` <br>
`make` <br>

Target Surface:<br>
`mkdir build-Target_Surface` <br>
`cd build-Target_Surface` <br>
`cmake ../Target_Surface/` <br>
`make` <br>

Next, check for the existence of the dynamic library path environment variable (LD_LIBRARY_PATH) <br>
`echo $LD_LIBRARY_PATH`

If there is nothing to be displayed, add a default path value <br>
`export LD_LIBRARY_PATH=/usr/local/lib/`

If everything went well, you should be able to run Caustic_Design and Target_Surface by entering the command: <br>
`./Caustic_Design`
or
`./Target_Surface`

## HOW TO USE
### Compute Optimal Transport
- First create your two grayscale images. The first image is the target image. This is the image that you want your lens or mirror to project. The second image is the source image. That is what the projection currently looks like which is in most cases a blank white image.
- Open Caustic_Design and click `File -> Load Image`, then load your target image. Next, click `File -> Load Source Image`, then load your source image.
- To set the optimal transport resolution, click `Algorithm -> Set Parameters` and change the `Sites` from 5000 to any other value. The higher this number is, the better your results will be, but the longer the optimal transport and the other steps will take. A site count of between 10k and 100k should be sufficient. You should also increase the `Levels` choose a value such that 4^(#levels) is about equal to the sites count. But ive noticed that the program crashes with a level count higher than 6, so i ket it at 6.
- Now finally click on `Algorithm -> Compute Optimal Transport`, click yes, and now the software will calculate the optimal transport between the source image and target image. It will take a couple of hours to complete depending on your specified steps and resolution. When it finishes computing, it will ask you to store the .weights file, save this file. You should also save the .dat file which you can do by clicking `File -> Save Source DAT`.

### Compute Interpolation
- Once you computed the optimal transport, the next thing to do is create a mesh in blender that will be deformed to create your caustic surface. I got the best results creating a cube, scaling it on the x-axis by 0.2, open it in edit mode, select the face facing the positive x-axis, click edge->subdivide, enter the amount of subdivisions you want, more results in higher resolution, but is slower to compute, then triangulate all faces by clicking 'Face' in the above menu and 'Triangulate Faces'.
- Then export this mesh as obj but make sure you unselect `Write Normals`, `Include UVs`, and `Write Materials` under `Geometry`.
- Next, launch the Target_Surface program and click `File -> Load Model`, then load your exported 3d model from blender, check to see if the subdivided face is facing the positive x-axis, and click `File -> Save Vertices`, save this file for the next step.
- Next, go back to Caustic_Design (this can be a new instance but doesnt have to be), Click on `Algorithm -> Compute Interpolation`, it will ask you for the target image again (the image you want to project), pointset (perviously generated .dat), weights (previously generated .weights), and light origin points (vertices generated from the previous step).
- when the last file is loaded, it will start the interpolation process. This will take a very long time depending on the resolution of the mesh and the amount of sites from the optimal transport step.
- When it is finished, it will ask you to save the interpolated point set (.dat), this file will be used to generate the caustic surface in the next step.

### Generate the Caustic Surface
- Open a new instance of the Target_Surface program and load your model again, then click on `File -> Load Light-Ray Receiver Position` open the .dat file from the interpolation step, if you then rotate the view, you should see your caustic image apear on the white target plane. if this is the case, click `Algorithm -> Run Target Optimization`. This will modify the surface of the model so it will project the caustic image. This will take a while depending on how easy it is to solve the surface hightmap.
- When the surface solver is done, you can export the model by clicking `File -> Save Model`, save the file as .stl
- You now have your caustic surface computed. This can be fabricated by a CNC machine. You can also simulate the caustics in blender using LuxRender.

## HOW DOES IT WORK
Lets consider a rectangular grid of prisms. Collimated light enters the prism grid and exits the grid in directions determined by the geometry of the prisms, generally opposite to the xy tilt of the facet where the light exits from. So by changing xy tilt of the prism facets, we can control where each light ray is being directed to. Now with some maths using Snell's law we can design a prism array that casts a shape using dots. Here is an example point set generated from the olympic rings.
![image](https://github.com/dylanmsu/Caustic-Design/assets/16963581/e599543f-e06e-473f-a4e4-54157bd38812)

We can generate a prism grid where each prism redirects the licht to one of the points of the point set. And thus creating an object that turns colimated licht into an olympic rings image.

The problem is, this prism grid is very dificult to make, even more so with smaller prisms. To solve this we can use an initially flat surface that is shaped like a rectangular mesh with vertices and faces. When we move one of the vertices of this mesh slightly outside of the flat surface, the faces that are connected to that vertex wil change their tilt slightly (think of 3d moddeling). Now, we can then ask an optimization algorithm to solve the hights of the vertices such that the tilt of every face is as close as possible to the required tilt calculated by snells law. The optimization algorithm will then spit out the ideal hights of every vertex such that when we shine colimated light through it, it wil project our image.

But in reality, this isn't quite as simple as that. That's where optimal transport comes in...

To get the best results, we need neighboring facets to have as little as possible tilt difference. We cannot guarantee this if we assign each facet to each target point at random. In fact, this will generate very poor, or even no results.

TODO

