# dGraph
A python library of graphics classes for defining a scene graph by using render stacks.

The addition of render stacks allows for realtime compositing effects, such as HUDs, distortion warps, blurs, FFT, or anything else that could be written up as a shader.

## Requires
- OpenGL
- numpy
- ctypes (for rendering with uvs or normals and warp shaders)
- OpenCV (for textures and imageManipulation)
- GLFW (optional - for ui and tests)
