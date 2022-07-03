# Vulkan-N-Body-Simulation
Designed with the help of vulkan-tutorial.com, and my previous vulkan-render, I created an N-Body simulation. All of the physics computations and rendering are done on the GPU.


The Simulation is written in C++ and uses the Vulkan Library to communicate with the GPU. All physics equations came from my IB Physics course and are run in a compute shader. All rendering is done on the GPU as well. For window handling, the GLFW library is used. Currently, I'm working on an IMGUI implementation for the engine. Yes, yes, I know the code isn't very pretty. I had a great time working on this and it was super interesting to develop!!

Cool Features:
- Janky Physics
- 1000000 particles in realtime
- Velocity based colors?
