#!/usr/bin/python
'''Test for OBJ with material loader and shader

Petr Kellnhofer
June 2017

'''
__author__ = ('Petr Kellnhofer')
__version__ = '1.0'

import OpenGL
OpenGL.ERROR_CHECKING = True      # Uncomment for 2x speed up
OpenGL.ERROR_LOGGING = False       # Uncomment for speed up
#OpenGL.FULL_LOGGING = True         # Uncomment for verbose logging
#OpenGL.ERROR_ON_COPY = True        # Comment for release
import OpenGL.GL as GL
from OpenGL.GL import *
import math, os
import numpy as np
import sys; sys.path.append('..')
import dGraph as dg
import dGraph.ui as dgui
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.render as dgr
import dGraph.materials as dgm
import dGraph.shaders as dgshdr
import dGraph.lights as dgl
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph,file=None):                
    '''Load or create our sceneGraph'''
    scene = renderGraph.add(dg.SceneGraph('DoF_Scene', file))

    # Lights
    scene.ambientLight = np.array([1,1,1], np.float32) * 0.2
    scene.lights.append(dgl.PointLight(intensity = np.array([1,1,1], np.float32)*0.7, position = np.array([2,3,4], np.float32)))
    scene.lights.append(dgl.DirectionLight(intensity = np.array([1,1,1], np.float32)*0.7, direction = np.array([-1,0.2,1], np.float32)))
    

    # This guy has mtl and textures
    cube = scene.add(dgs.PolySurface('alien', scene, file = '%s/alien/alien.obj'%MODELDIR))
    cube.setScale(0.8,0.8,0.8)
    cube.setTranslate(0,-0.5,-2)
    cube.setRotate(0,0,0)

    

    # Tick
    camera = scene.add(dgc.Camera('scene', scene))
    camera.setResolution((renderGraph.width, renderGraph.height))
    camera.setTranslate(0.,0.,0.)
    camera.setFOV(50.)
    
    # Final
    renderGraph.frameBuffer.connectInput(camera)


    return True                                                         # Initialization Successful

def animateScene(renderGraph, frame):
    # infinity rotate:
    y = math.sin(frame*math.pi/60)
    x = math.cos(frame*math.pi/30)/4
    for scene in renderGraph.scenes:
        for obj in renderGraph[scene].shapes:
            renderGraph[scene][obj].rotate += np.array((x,y,0.))
    # update focus:
    
def addInput(renderGraph):
    dgui.add_key_callback(arrowKey, dgui.KEY_RIGHT, renderGraph=renderGraph, direction=3)
    dgui.add_key_callback(arrowKey, dgui.KEY_LEFT, renderGraph=renderGraph, direction=2)
    dgui.add_key_callback(arrowKey, dgui.KEY_UP, renderGraph=renderGraph, direction=1)
    dgui.add_key_callback(arrowKey, dgui.KEY_DOWN, renderGraph=renderGraph, direction=0)

def arrowKey(window,renderGraph,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        pass
    else:                   # print "down"
        pass

def setup():
    dgui.init()
    renderGraph = dgr.RenderGraph('TestMTL_RG')
    monitors = dgui.get_monitors()
    display = renderGraph.add(dgui.Display('Last',monitors[-1]))
    offset = display.screenPosition
    mainWindow = renderGraph.add(dgui.open_window('Render Graph Test', offset[0], offset[1], display.width, display.height))
    if not mainWindow:
        dgui.terminate()
        exit(1)
    x, y = dgui.get_window_pos(mainWindow)
    width, height = dgui.get_window_size(mainWindow)
    dgui.add_key_callback(dgui.close_window, dgui.KEY_ESCAPE)
    dg.initGL()
    scene = loadScene(renderGraph)
    renderGraph.graphicsCardInit()
    return renderGraph, [mainWindow]

def runLoop(renderGraph, mainWindow):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    frame = 0
    totalSleep = 0
    start = time.time()
    while not dgui.window_should_close(mainWindow):
        dgui.make_context_current(mainWindow)
        renderGraph.render()
        now = time.time()
        toSleep = max(0,(frame+1)/config.maxFPS+start-now)
        time.sleep(toSleep)
        dgui.swap_buffers(mainWindow)
        dgui.poll_events()
        animateScene(renderGraph, frame)
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
    renderGraph, windows = setup()
    addInput(renderGraph)
    runLoop(renderGraph, windows[0])
