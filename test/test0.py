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
import dGraph.ui as ui
import dGraph.render as dgr
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.materials as dgm
import dGraph.shaders as dgshdr
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph,file=None):
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph(file)
    cam = dgc.Camera('cam', scene)
    cam.setResolution((renderGraph.width, renderGraph.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    renderGraph.cameras.append(cam)
    #teapot = dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR)
    teapot = dgs.PolySurface('teapot', scene, file = '%s/octoAlien.obj'%MODELDIR)
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(0.,-.20,-1.)
    teapot.setRotate(0.,0.,0.)
    renderGraph.objects['teapot'] = teapot

    #material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    material1 = dgm.Lambert('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in renderGraph.objects.values():
        obj.setMaterial(material1)

    renderGraph.append(cam)
    #warp = dgsdr.Lookup('lookup1',lutFile='%s/warp_0020.npy'%MODELDIR)
    #renderGraph.append(warp)
    return scene

def animateScene(renderGraph, frame):
    ''' Create motion in our scene '''
    # infinity rotate:
    y = 1
    x = math.cos(frame*math.pi/60)
    for obj in renderGraph.objects.values():
        obj.rotate += np.array((x,y,0.))
    
def drawGLScene(renderGraph):
    ''' Draw everything in renderGraph '''
    myStack = list(renderGraph)                                     # copy the renderGraph so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderGraph.width, renderGraph.height, myStack)     # Render our camera to screen

def addInput(renderGraph):
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderGraph=renderGraph, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderGraph=renderGraph, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderGraph=renderGraph, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderGraph=renderGraph, direction=0)

''' In order to do input with mouse and keyboard, we need to setup a state machine with 
states switches from mouse presses and releases and certain key presses 
- maybe enable a glfwSetCursorPosCallback when button is pressed, or
- more likely just poll cursor position since you can't disable a callback'''
def arrowKey(window,renderGraph,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        renderGraph.objects['teapot'].rotate += np.array((5.,0.,0.))
        print(renderGraph.objects['teapot'].rotate)
    else:                   # print "down"
        renderGraph.objects['teapot'].rotate -= np.array((5.,0.,0.))
        print(renderGraph.objects['teapot'].rotate)

def drawScene(renderGraph):
    ''' Render the stack '''
    myStack = list(renderGraph)                                     # copy the renderGraph so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderGraph.width, renderGraph.height, myStack)     # Render our warp to screen

def setup():
    renderGraph = ui.RenderGraph()
    renderGraph.displays.append(ui.Display(resolution=(800,600)))
    ui.init()
    offset = (0,0)
    mainWindow = renderGraph.addWindow(ui.open_window('Scene Graph Test', offset[0], offset[1], renderGraph.displays[0].width, renderGraph.displays[0].height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    x, y = ui.get_window_pos(mainWindow)
    width, height = ui.get_window_size(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    dg.initGL()
    scene = loadScene(renderGraph)
    renderGraph.graphicsCardInit()
    return renderGraph, scene, [mainWindow]

def runLoop(renderGraph, mainWindow):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    frame = 0
    start = time.time()
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        drawScene(renderGraph)
        now = time.time()
        time.sleep((frame+1)/config.maxFPS+start-now)
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        animateScene(renderGraph, frame)
        frame += 1
    ui.terminate()
    exit(0)

if __name__ == '__main__':
    renderGraph, scene, windows = setup()
    addInput(renderGraph)
    runLoop(renderGraph, windows[0])
