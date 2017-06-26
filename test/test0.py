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
import dGraph.render as dgr
import dGraph.cameras as dgc
import dGraph.shapes as dgs
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

    material1 = scene.add(dgm.Material('material1'))
    material1.diffuseColor *= 0.4
    for obj in scene.shapes:
        scene[obj].setMaterial(material1)

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

''' In order to do input with mouse and keyboard, we need to setup a state machine with 
states switches from mouse presses and releases and certain key presses 
- maybe enable a glfwSetCursorPosCallback when button is pressed, or
- more likely just poll cursor position since you can't disable a callback'''
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

def setup():
    renderGraph = dgr.RenderGraph('Test0_RG')
    display = renderGraph.add(dgui.Display('Fake Display',resolution=(800,600)))
    dgui.init()
    offset = (0,0)
    mainWindow = renderGraph.add(dgui.open_window('Scene Graph Test', offset[0], offset[1], display.width, display.height))
    if not mainWindow:
        dgui.terminate()
        exit(1)
    x, y = dgui.get_window_pos(mainWindow)
    width, height = dgui.get_window_size(mainWindow)
    dgui.add_key_callback(dgui.close_window, dgui.KEY_ESCAPE)
    dg.initGL()
    scene = loadScene(renderGraph)
    renderGraph.graphicsCardInit()
    return scene, [mainWindow]

def runLoop(scene, mainWindow):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    frame = 0
    totalSleep = 0
    start = time.time()
    while not dgui.window_should_close(mainWindow):
        dgui.make_context_current(mainWindow)
        scene.render()
        now = time.time()
        toSleep = max(0,(frame+1)/config.maxFPS+start-now)
        time.sleep(toSleep)
        dgui.swap_buffers(mainWindow)
        dgui.poll_events()
        animateScene(scene, frame)
        totalSleep += toSleep
        frame += 1
    end = time.time()
    dgui.terminate()
    elapsed = end-start
    computePct = (1-totalSleep/elapsed)*100
    renderTime = elapsed-totalSleep
    frameTime = renderTime/frame*1000
    print('Average frame took %.4f ms to render.\nRendered %.4f seconds of a total %.4f seconds.\nRendering %.2f%% of the time.'%(frameTime,renderTime,elapsed,computePct))
    exit(0)

if __name__ == '__main__':
    scene, windows = setup()
    addInput(scene)
    runLoop(scene, windows[0])
