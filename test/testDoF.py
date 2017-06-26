#!/usr/bin/python
'''Test for faster DoF renderer

Petr Kellnhofer
June 2017 - copied from test2.py by David Dunn

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
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph,file=None):                
    '''Load or create our sceneGraph'''
    scene = renderGraph.add(dg.SceneGraph('DoF_Scene', file))

    cube = scene.add(dgs.PolySurface('cube', scene, file = '%s/TexturedCube.obj'%MODELDIR))
    cube.setScale(1,1,1)
    cube.setTranslate(0.,0.,-4)
    cube.setRotate(25.,65.,23.)
    
    teapot = scene.add(dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR))
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(.5,-.2,-1.5)
    teapot.setRotate(5.,0.,0.)
    teapot.material.diffuseColor = np.array([1.0, 1.0, 0.8]) * 0.5
    teapot.material.specularColor = np.array([1.0, 0.8, 0.7]) * 0.5
    teapot.material.glossiness = 20
    
    

    renderGraph.focus = 2.
    renderGraph.focusChanged = False
    display = renderGraph['Fake Display']
    imageScale = display.width/(renderGraph.width)
    pixelDiameter = imageScale*display.pixelSize()[0]

    # Tick
    camera = scene.add(dgc.Camera('scene', scene))
    camera.setResolution((renderGraph.width, renderGraph.height))
    camera.setTranslate(0.,0.,0.)
    camera.setFOV(50.)
    # Tock
    camBuffer = renderGraph.add(dgr.FrameBuffer('camera_fbo',onScreen=False,depthIsTexture=True))
    camBuffer.connectInput(camera)

    # Tick
    gaussMip = renderGraph.add(dgshdr.GaussMIPMap('imageGaussMip'))
    gaussMip.connectInput(camBuffer.rgba, 'inputImage')
    gaussMip._width =  display.width
    gaussMip._height = display.height # This should not be called here but we need to to be able to call gaussMip.mipLevelCount) - design flaw?
    # Tock
    gaussMipBuffer = renderGraph.add(dgr.FrameBuffer('gaussMip_fbo',onScreen=False,mipLevels=gaussMip.mipLevelCount))
    gaussMipBuffer.connectInput(gaussMip)

    # Tick
    glDepthToLinear = renderGraph.add(dgshdr.GLDepthToLinear('glDepthToLinear'))
    glDepthToLinear.inputRange[1] = 1 # The depth scaling is not necessary if one assumes the world is properly modeled with units in meters
    glDepthToLinear.outputRange[1] = 1#0.1
    glDepthToLinear.projectionMatrix = camera.filmMatrix
    glDepthToLinear.connectInput(camBuffer.depth, 'glDepthTexture2D')

    # Tock
    linDepthBuffer = renderGraph.add(dgr.FrameBuffer('linDepth_fbo',onScreen=False,isf=GL_R32F))
    linDepthBuffer.connectInput(glDepthToLinear)

    # Tick
    dof = renderGraph.add(dgshdr.DepthOfField('depthOfField'))
    renderGraph.focusDistanceLog = 0.1
    dof.focalPlaneMeters = 10**renderGraph.focusDistanceLog
    print("Current focal depth = %.3f D / %.2f meters" % (1.0/renderGraph['depthOfField'].focalPlaneMeters, renderGraph['depthOfField'].focalPlaneMeters))
    dof.pixelSizeMm = pixelDiameter * 1e3
    dof.apertureMm = 3.0 * 10
    dof.connectInput(gaussMipBuffer.rgba, 'imageTexture')
    dof.connectInput(linDepthBuffer.rgba, 'depthTexture')

    # Final
    renderGraph.frameBuffer.connectInput(dof)


    return True                                                         # Initialization Successful

def animateScene(renderGraph, frame):
    # infinity rotate:
    y = math.sin(frame*math.pi/60)
    x = math.cos(frame*math.pi/30)/4
    for scene in renderGraph.scenes:
        for obj in renderGraph[scene].shapes:
            renderGraph[scene][obj].rotate += np.array((x,y,0.))
    # update focus:
    if renderGraph.focusChanged:
        renderGraph['depthOfField'].focalPlaneMeters = 10**renderGraph.focusDistanceLog
        print("Current focal depth = %.3f D / %.2f meters" % (1.0/renderGraph['depthOfField'].focalPlaneMeters, renderGraph['depthOfField'].focalPlaneMeters))
        renderGraph.focusChanged = False
    
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
        renderGraph.focusDistanceLog += 0.1
        renderGraph.focusChanged = True
    else:                   # print "down"
        renderGraph.focusDistanceLog -= 0.1
        renderGraph.focusChanged = True

def setup():
    renderGraph = dgr.RenderGraph('Test2_RG')
    display = renderGraph.add(dgui.Display('Fake Display',resolution=(1920,1200),size=(.518,.324)))
    dgui.init()
    offset = (-1920,0)
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
