#!/usr/bin/python
'''Test for an openGL based stereo renderer - test distortion warp texture

David Dunn
Feb 2017 - created

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.0'

import OpenGL
OpenGL.ERROR_CHECKING = False      # Uncomment for 2x speed up
OpenGL.ERROR_LOGGING = False       # Uncomment for speed up
#OpenGL.FULL_LOGGING = True         # Uncomment for verbose logging
#OpenGL.ERROR_ON_COPY = True        # Comment for release
import OpenGL.GL as GL
import cv2
import numpy as np
import dDisplay as dd
import dGraph as dg
import dGraph.ui as ui
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.materials as dgm
import dGraph.materials.warp
import dGraph.util.imageManip as im

modelDir = './dGraph/test/data'

def loadScene(renderStack,file=None):                
    '''Load or create our sceneGraph'''
    
    scene = dg.SceneGraph(file)
    cam = dgc.Camera('cam', scene)
    cam.setResolution((renderStack.width, renderStack.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    renderStack.cameras.append(cam)
    teapot = dgs.PolySurface('teapot', backScene, file = '%s/teapot.obj'%modelDir)
    teapot.setScale(.1,.1,.1)
    teapot.setTranslate(.0,-.05,-2.)
    teapot.setRotate(5.,0.,0.)
    renderStack.objects['teapot'] = teapot