#!/usr/bin/python
'''Test for an openGL based stereo renderer - test texture mapping

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
import os
import numpy as np
import dGraph as dg
import dGraph.ui as dgui
import dGraph.test as dgtest
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.materials as dgm
import dGraph.lights as dgl
import dGraph.shaders as dgshdr
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph):                
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph('Test3_SG')
    cam = scene.add(dgc.StereoCamera('cam', scene))
    cam.setResolution((renderGraph.width, renderGraph.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    
    teapot = scene.add(dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR))
    teapot.setScale(.35,.35,.35)
    teapot.setTranslate(-.02,.02,-2.)
    teapot.setRotate(5.,-15.,0.)

    material1 = scene.add(dgm.Material('material1'))
    for obj in scene.shapes:
        scene[obj].setMaterial(material1)

    scene.ambientLight = np.array([1,1,1], np.float32) * 0.2
    scene.lights.append(dgl.PointLight(intensity = (0,1,1), position = (2,3,4)))
    scene.lights.append(dgl.DirectionLight(intensity = (1,0,1), direction = (-1,0.5,0.1)))

    renderGraph.frameBuffer.connectInput(cam.left, posWidth=0, posHeight=0, width=.5, height=1)
    renderGraph.frameBuffer.connectInput(cam.right, posWidth=.5, posHeight=0, width=.5, height=1)
    scene.add(renderGraph)
    #warp = dgm.warp.Lookup('lookup1',lutFile='%s/warp_0020.npy'%MODELDIR)
    #renderStack.append(warp)
    return scene

def addInput(renderStack):
    dgui.add_key_callback(arrowKey, dgui.KEY_RIGHT, renderStack=renderStack, direction=3)
    dgui.add_key_callback(arrowKey, dgui.KEY_LEFT, renderStack=renderStack, direction=2)
    dgui.add_key_callback(arrowKey, dgui.KEY_UP, renderStack=renderStack, direction=1)
    dgui.add_key_callback(arrowKey, dgui.KEY_DOWN, renderStack=renderStack, direction=0)

def arrowKey(window,renderStack,direction):
    rotate = np.array((0.,0.,0.))
    translate = np.array((0.,0.,0.))
    if direction == 3:    # print "right"
        rotate += np.array((0.,5.,0.))
    elif direction == 2:    # print "left"
        rotate -= np.array((0.,5.,0.))
    elif direction == 1:      # print 'up'
        translate += np.array((0.,.01,0.))
    else:                   # print "down"
        translate -= np.array((0.,.01,0.))
    for obj in scene.shapes:
        scene[obj].rotate += rotate
        scene[obj].translate += translate
'''
def drawScene(renderStack):
    ' '' Render the stack ' ''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our warp to screen

def setup():
    renderStack = ui.RenderStack()
    renderStack.display = ui.Display()
    ui.init()
    offset = (0,0)
    mainWindow = renderStack.addWindow(ui.open_window('Warp Distortion Test', offset[0], offset[1], renderStack.display.width, renderStack.display.height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    x, y = ui.get_window_pos(mainWindow)
    width, height = ui.get_window_size(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    dg.initGL()
    scene = loadScene(renderStack)
    renderStack.graphicsCardInit()
    return renderStack, scene, [mainWindow]

def runLoop(renderStack, mainWindow):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    start = time.time()
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        drawScene(renderStack)
        now = time.time()
        time.sleep(max((frame+1)/config.maxFPS+start-now,0))
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        #ui.wait_events()
    ui.terminate()
    exit(0)
'''

if __name__ == '__main__':
    scene, windows = dgtest.setup(loadScene)
    addInput(scene)
    print("Hit ESC key to quit.")
    dgtest.runLoop(scene, windows[0])
