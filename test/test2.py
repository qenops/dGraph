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
import dGraph.materials.warp
import dGraph.util.imageManip as im

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderStack,file=None):                
    '''Load or create our sceneGraph'''
    backScene = dg.SceneGraph(file)
    bCam = dgc.Camera('back', backScene)
    bCam.setResolution((renderStack.width, renderStack.height))
    bCam.setTranslate(0.,0.,0.)
    bCam.setFOV(50.)
    renderStack.cameras.append(bCam)
    cube = dgs.PolySurface('cube', backScene, file = '%s/cube.obj'%MODELDIR)
    cube.setScale(.4,.4,.4)
    cube.setTranslate(0.,0.,-2.)
    cube.setRotate(25.,65.,23.)
    renderStack.objects['cube'] = cube
    
    frontScene = dg.SceneGraph(file)
    fCam = dgc.Camera('front', frontScene)
    fCam.setResolution((renderStack.width, renderStack.height))
    fCam.translate.connect(bCam.translate)
    fCam.rotate.connect(bCam.rotate)
    fCam.setFOV(50.)
    teapot = dgs.PolySurface('teapot', frontScene, file = '%s/teapot.obj'%MODELDIR)
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(.5,-.2,-1.5)
    teapot.setRotate(5.,0.,0.)
    renderStack.objects['teapot'] = teapot
    
    material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in renderStack.objects.itervalues():
        obj.setMaterial(material1)
    
    renderStack.focus = 2.
    renderStack.focusChanged = False
    imageScale = renderStack.displays[0].width/(renderStack.width)
    pixelDiameter = imageScale*renderStack.displays[0].pixelSize()[0]
    kernel = im.getPSF(renderStack.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
    ftBlur = dgm.warp.Convolution('ftBlur')
    ftBlur.kernel = kernel
    renderStack.shaders['ftBlur'] = ftBlur
    ftBlur._width = renderStack.width
    ftBlur._height = renderStack.height
    print ftBlur.fragmentShader
    
    bkBlur = dgm.warp.Convolution('bkBlur')
    kernel = im.getPSF(renderStack.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
    bkBlur.kernel = kernel
    renderStack.shaders['bkBlur'] = bkBlur

    over = dgm.warp.Over('over')
    over.overStackAppend(fCam)
    over.overStackAppend(ftBlur)
    over.underStackAppend(bCam)
    over.underStackAppend(bkBlur)
    renderStack.append(over)
    return True                                                         # Initialization Successful

def animateScene(renderStack, frame):
    # infinity rotate:
    y = math.sin(frame*math.pi/60)
    x = math.cos(frame*math.pi/30)/4
    for obj in renderStack.objects.itervalues():
        obj.rotate += np.array((x,y,0.))
    # update focus:
    if renderStack.focusChanged:
        imageScale = renderStack.displays[0].width/(renderStack.width)
        pixelDiameter = imageScale*renderStack.displays[0].pixelSize()[0]
        kernel = im.getPSF(renderStack.focus, 1.5, aperture=.004, pixelDiameter=pixelDiameter)
        renderStack.shaders['ftBlur'].kernel = kernel
        kernel = im.getPSF(renderStack.focus, 2., aperture=.004, pixelDiameter=pixelDiameter)
        renderStack.shaders['bkBlur'].kernel = kernel
        renderStack.focusChanged = False
    
def addInput(renderStack):
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=renderStack, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=renderStack, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=renderStack, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=renderStack, direction=0)

def arrowKey(window,renderStack,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        renderStack.focus += .1
        renderStack.focusChanged = True
        print "Current focal depth = %s"%renderStack.focus
    else:                   # print "down"
        renderStack.focus -= .1
        renderStack.focusChanged = True
        print "Current focal depth = %s"%renderStack.focus

def drawScene(renderStack):
    ''' Draw everything in renderStack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our stack to screen

def setup():
    renderStack = ui.RenderStack()
    renderStack.displays.append(ui.Display(resolution=(1920,1200),size=(.518,.324)))
    ui.init()
    offset = (1920,0)
    mainWindow = renderStack.addWindow(ui.open_window('Render Stack Test', offset[0], offset[1], renderStack.displays[0].width, renderStack.displays[0].height))
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
    print("Use Up/Down to change focal depth.")
    frame = 0
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        drawScene(renderStack)
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        animateScene(renderStack, frame)
        frame += 1
    ui.terminate()
    exit(0)

if __name__ == '__main__':
    renderStack, scene, windows = setup()
    addInput(renderStack)
    runLoop(renderStack, windows[0])
