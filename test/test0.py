#!/usr/bin/python
'''Test for an openGL based renderer - testing basic openGL rasteriziation

David Dunn
Feb 2017 - created test suite

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
import math, os
import numpy as np
import dGraph as dg
import dGraph.ui as dgui
import dGraph.test as dgtest
import dGraph.render as dgr
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.lights as dgl
import dGraph.materials as dgm
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph):
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph('Test0_SG')
    cam = scene.add(dgc.Camera('cam', scene))
    cam.setResolution((renderGraph.width, renderGraph.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    #teapot = scene.add(dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR))
    teapot = scene.add(dgs.PolySurface('teapot', scene, file = '%s/octoAlien.obj'%MODELDIR))
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(0.,-.20,-1.)
    teapot.setRotate(0.,0.,0.)
    # Materials
    material1 = scene.add(dgm.Material('material1'))
    material1.diffuseColor *= 0.4
    for obj in scene.shapes:
        scene[obj].setMaterial(material1)
    # Lights
    scene.ambientLight = np.array([1,1,1], np.float32) * 0.2
    scene.lights.append(dgl.PointLight(intensity = (0,1,1), position = (2,3,4)))
    scene.lights.append(dgl.DirectionLight(intensity = (1,0,1), direction = (-1,0.5,0.1)))
    # Animate
    scene.animateFunc = animateScene
    # Render
    renderGraph.frameBuffer.connectInput(cam)
    scene.add(renderGraph)
    return scene

def animateScene(scene, frame):
    ''' Create motion in our scene '''
    # infinity rotate:
    y = 1
    x = math.cos(frame*math.pi/60)
    for obj in scene.shapes:
        scene[obj].rotate += np.array((x,y,0.))

def addInput(scene):
    dgui.add_key_callback(arrowKey, dgui.KEY_RIGHT, scene=scene, direction=3)
    dgui.add_key_callback(arrowKey, dgui.KEY_LEFT, scene=scene, direction=2)
    dgui.add_key_callback(arrowKey, dgui.KEY_UP, scene=scene, direction=1)
    dgui.add_key_callback(arrowKey, dgui.KEY_DOWN, scene=scene, direction=0)

def arrowKey(window,scene,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        scene['teapot'].rotate += np.array((5.,0.,0.))
        print(scene['teapot'].rotate)
    else:                   # print "down"
        scene['teapot'].rotate -= np.array((5.,0.,0.))
        print(scene['teapot'].rotate)

if __name__ == '__main__':
    scene, windows = dgtest.setup(loadScene)
    addInput(scene)
    print("Hit ESC key to quit.")
    dgtest.runLoop(scene, windows[0])
