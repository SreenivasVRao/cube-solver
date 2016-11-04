# Cube Solver
Implements a Rubik's cube colour detection system on top of existing Kociemba algorithm implementation and 3D model visualization


#Installation instructions

`sudo pip install kociemba`
`git clone https://github.com/SreenivasVRao/cube-solver.git`
`cd cube-solver`
`git clone https://github.com/SreenivasVRao/MagicCube.git`
`python test.py`


#Usage Notes

1. System requires uniform lighting, preferably overhead.
2. Show the Rubik's cube to the camera in the following order of faces: 
Front,Back, Right, Left, Up, Down. (This corresponds to White, Yellow, Green, Blue, Red, Orange in the code.)
3. Note that the orientation of the faces with respect to each other must be maintained. To do so, rotate the cube on only one axis at a time.
4. Implementation is specific to the 3x3x3 Hungarian Horror!

#Acknowledgements
Thank you to @davidwhogg and @jakevdp for the MagicCube repo. Thanks as well to @muodov for the Kociemba algorithm implementation.

#LICENSE
All content copyright 2016 respective authors. Cube Solver is licensed under the GPLv3 License. See `LICENSE.txt` for more information.



