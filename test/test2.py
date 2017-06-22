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
import dGraph.ui as ui
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
    backScene = dg.SceneGraph(file)
    bCam = backScene.add(dgc.Camera('back', backScene))
    bCam.setResolution((renderGraph.width, renderGraph.height))
    bCam.setTranslate(0.,0.,0.)
    bCam.setFOV(50.)
    cube = backScene.add(dgs.PolySurface('cube', backScene, file = '%s/cube.obj'%MODELDIR))
    cube.setScale(.4,.4,.4)
    cube.setTranslate(0.,0.,-2.)
    cube.setRotate(25.,65.,23.)
    renderGraph.scenes.append(backScene)
    
    frontScene = dg.SceneGraph(file)
    fCam = frontScene.add(dgc.Camera('front', frontScene))
    fCam.setResolution((renderGraph.width, renderGraph.height))
    fCam.translate.connect(bCam.translate)
    fCam.rotate.connect(bCam.rotate)
    fCam.setFOV(50.)
    teapot = frontScene.add(dgs.PolySurface('teapot', frontScene, file = '%s/teapot.obj'%MODELDIR))
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(.5,-.2,-1.5)
    teapot.setRotate(5.,0.,0.)
    renderGraph.scenes.append(frontScene)
    
    material1 = backScene.add(frontScene.add(dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)))
    for scene in renderGraph.scenes:
        for obj in scene.values():
            if isinstance(obj, dgs.Shape):
                obj.setMaterial(material1)
    
    renderGraph.focus = 2.
    renderGraph.focusChanged = False
    imageScale = renderGraph.displays[0].width/(renderGraph.width)
    pixelDiameter = imageScale*renderGraph.displays[0].pixelSize()[0]
    kernel = im.getPSF(renderGraph.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
    ftBlur = dgshdr.Convolution('ftBlur')
    ftBlur.kernel = kernel
    renderGraph.shaders['ftBlur'] = ftBlur
    ftBlur._width = renderGraph.width
    ftBlur._height = renderGraph.height
    
    bkBlur = dgshdr.Convolution('bkBlur')
    kernel = im.getPSF(renderGraph.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
    bkBlur.kernel = kernel
    renderGraph.shaders['bkBlur'] = bkBlur

    over = dgshdr.Over('over')
    over.overStackAppend(fCam)
    over.overStackAppend(ftBlur)
    over.underStackAppend(bCam)
    over.underStackAppend(bkBlur)
    renderGraph.append(over)
    return True                                                         # Initialization Successful

def animateScene(renderGraph, frame):
    # infinity rotate:
    y = math.sin(frame*math.pi/60)
    x = math.cos(frame*math.pi/30)/4
    for obj in renderGraph.objects.values():
        obj.rotate += np.array((x,y,0.))
    # update focus:
    if renderGraph.focusChanged:
        imageScale = renderGraph.displays[0].width/(renderGraph.width)
        pixelDiameter = imageScale*renderGraph.displays[0].pixelSize()[0]
        kernel = im.getPSF(renderGraph.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
        renderGraph.shaders['ftBlur'].kernel = kernel
        kernel = im.getPSF(renderGraph.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
        renderGraph.shaders['bkBlur'].kernel = kernel
        renderGraph.focusChanged = False
    
def addInput(renderGraph):
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderGraph=renderGraph, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderGraph=renderGraph, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderGraph=renderGraph, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderGraph=renderGraph, direction=0)

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

def drawScene(renderGraph):
    ''' Draw everything in renderGraph '''
    myStack = list(renderGraph)                                     # copy the renderGraph so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderGraph.width, renderGraph.height, myStack)     # Render our stack to screen

def setup():
    renderGraph = ui.RenderGraph()
    renderGraph.displays.append(ui.Display(resolution=(1920,1200),size=(.518,.324)))
    ui.init()
    offset = (0,0)
    mainWindow = renderGraph.addWindow(ui.open_window('Render Stack Test', offset[0], offset[1], renderGraph.displays[0].width, renderGraph.displays[0].height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    x, y = ui.get_window_pos(mainWindow)
    width, height = ui.get_window_size(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
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
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        scene.render()
        now = time.time()
        toSleep = max(0,(frame+1)/config.maxFPS+start-now)
        time.sleep(toSleep)
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        animateScene(scene, frame)
        totalSleep += toSleep
        frame += 1
    end = time.time()
    ui.terminate()
    elapsed = end-start
    computePct = (1-totalSleep/elapsed)*100
    print('Slept %.4f seconds of a total %.4f seconds.\nRendering %.2f%% of the time.'%(totalSleep,elapsed,computePct))
    exit(0)

if __name__ == '__main__':
    scene, windows = setup()
    addInput(scene)
    runLoop(scene, windows[0])

