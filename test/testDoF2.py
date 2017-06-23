#!/usr/bin/python
'''Test for an openGL based stereo renderer - testing render stack features (compositing via over shader and blurring via convolution shader)

David Dunn
Feb 2017 - converted to test suite - failing depth test for render to framebuffers

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
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.render as dgr
import dGraph.materials as dgm
import dGraph.shaders as dgshdr
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph):                
    '''Load or create our sceneGraph'''
    
    backScene = renderGraph.add(dg.SceneGraph('Test2_backSG'))
    bCam = backScene.add(dgc.Camera('back', backScene))
    bCam.setResolution((renderGraph.width, renderGraph.height))
    bCam.setTranslate(0.,0.,0.)
    bCam.setFOV(50.)
    cube = backScene.add(dgs.PolySurface('cube', backScene, file = '%s/cube.obj'%MODELDIR))
    cube.setScale(.4,.4,.4)
    cube.setTranslate(0.,0.,-2.)
    cube.setRotate(25.,65.,23.)
    
    frontScene = renderGraph.add(dg.SceneGraph('Test2_frontSG'))
    fCam = frontScene.add(dgc.Camera('front', frontScene))
    fCam.setResolution((renderGraph.width, renderGraph.height))
    fCam.translate.connect(bCam.translate)
    fCam.rotate.connect(bCam.rotate)
    fCam.setFOV(50.)
    teapot = frontScene.add(dgs.PolySurface('teapot', frontScene, file = '%s/teapot.obj'%MODELDIR))
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(.5,-.2,-1.5)
    teapot.setRotate(5.,0.,0.)
    
    material1 = backScene.add(frontScene.add(dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)))
    for scene in renderGraph.scenes:
        for obj in renderGraph[scene].shapes:
            renderGraph[scene][obj].setMaterial(material1)
    
    ftBuffer = renderGraph.add(dgr.FrameBuffer('ftFB',onScreen=False))
    ftBuffer.connectInput(fCam)
    bkBuffer = renderGraph.add(dgr.FrameBuffer('bkFB',onScreen=False))
    bkBuffer.connectInput(bCam)

    renderGraph.focus = 2.  # should this be property of camera?
    renderGraph.focusChanged = False
    display = renderGraph['Fake Display']
    imageScale = display.width/(renderGraph.width)
    pixelDiameter = imageScale*display.pixelSize()[0]
    kernel = im.getPSF(renderGraph.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
    ftBlur = renderGraph.add(dgshdr.Convolution('ftBlur'))
    ftBlur.kernel = kernel
    ftBlur.connectInput(ftBuffer.rgba, 'texRGBA')
    bkBlur = renderGraph.add(dgshdr.Convolution('bkBlur'))
    kernel = im.getPSF(renderGraph.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
    bkBlur.kernel = kernel
    bkBlur.connectInput(bkBuffer.rgba, 'texRGBA')

    ftBlurBuffer = renderGraph.add(dgr.FrameBuffer('ftBlurFB',onScreen=False))
    ftBlurBuffer.connectInput(ftBlur)
    bkBlurBuffer = renderGraph.add(dgr.FrameBuffer('bkBlurFB',onScreen=False))
    bkBlurBuffer.connectInput(bkBlur)

    over = renderGraph.add(dgshdr.Over('over'))
    over.connectInput(ftBlurBuffer.rgba, 'texOVER_RGBA')
    over.connectInput(bkBlurBuffer.rgba, 'texUNDER_RGBA')
    renderGraph.frameBuffer.connectInput(over)
    
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
        display = renderGraph['Fake Display']
        imageScale = display.width/(renderGraph.width)
        pixelDiameter = imageScale*display.pixelSize()[0]
        kernel = im.getPSF(renderGraph.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
        renderGraph['ftBlur'].kernel = kernel
        kernel = im.getPSF(renderGraph.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
        renderGraph['bkBlur'].kernel = kernel
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
        renderGraph.focus += .1
        renderGraph.focusChanged = True
        print("Current focal depth = %s"%renderGraph.focus)
    else:                   # print "down"
        renderGraph.focus -= .1
        renderGraph.focusChanged = True
        print("Current focal depth = %s"%renderGraph.focus)

def setup():
    renderGraph = dgr.RenderGraph('Test2_RG')
    display = renderGraph.add(dgui.Display('Fake Display',resolution=(1920,1200),size=(.518,.324)))
    dgui.init()
    offset = (0,0)
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

