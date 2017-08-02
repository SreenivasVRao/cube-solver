# Cube Solver
Implements a Rubik's cube colour detection system on top of existing Kociemba algorithm implementation and 3D model visualization


# Installation instructions

    sudo pip install kociemba
    git clone https://github.com/SreenivasVRao/cube-solver.git
    cd cube-solver
    git clone https://github.com/SreenivasVRao/MagicCube.git


To test the system, run:

    python test.py


# Usage Notes

1. System requires uniform lighting, preferably overhead.
2. Show the Rubik's cube to the camera in the following order of faces: 
Front,Back, Right, Left, Up, Down. (This corresponds to White, Yellow, Green, Blue, Red, Orange in the code.)
3. Note that the orientation of the faces with respect to each other must be maintained. To do so, rotate the cube on only one axis at a time.
4. Implementation is specific to the 3x3x3 Hungarian Horror!


# Examples

A sample set of inputs and outputs is available in `photos/inputs` and `photos/outputs`.

A video showing the visualization and solution of the cube is available here: https://youtu.be/Jx_qR5r4UQ8

The visualization is a modification of https://github.com/davidwhogg/MagicCube

I have added an option to reverse engineer the scramble so that you can run the simulation multiple times with the "rescramble" button. 

"Solve Cube" calls the solver built by David Hogg. It's unclear what method it is using, but it is probably similar to the Kociemba algorithm.

"Alternate Solution" is an added functionality that explicitly uses the Kociemba algorithm to solve it. It also runs slower than the "Solve cube" button, so you can follow the solution on your own cube.

In the terminal, you can see the scramble and the solution in Singmaster notation.

# Acknowledgements
Thank you to @davidwhogg and @jakevdp for the MagicCube repo. Thanks as well to @muodov for the Kociemba algorithm implementation.

# LICENSE
All content copyright 2016 respective authors. Cube Solver is licensed under the GPLv3 License. See `LICENSE.txt` for more information.



