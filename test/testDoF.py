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
import math, os
import numpy as np
import sys; sys.path.append('..')
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

def loadScene(renderStack,file=None):                
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph(file)
    camera = dgc.Camera('cene', scene)
    camera.setResolution((renderStack.width, renderStack.height))
    camera.setTranslate(0.,0.,0.)
    camera.setFOV(50.)
    renderStack.cameras.append(camera)

    cube = dgs.PolySurface('cube', scene, file = '%s/cube.obj'%MODELDIR)
    cube.setScale(.4,.4,.4)
    cube.setTranslate(0.,0.,-2.)
    cube.setRotate(25.,65.,23.)
    renderStack.objects['cube'] = cube
    
    teapot = dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR)
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(.5,-.2,-1.5)
    teapot.setRotate(5.,0.,0.)
    renderStack.objects['teapot'] = teapot
    
    material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in renderStack.objects.values():
        obj.setMaterial(material1)
    
    renderStack.focus = 2.
    renderStack.focusChanged = False
    imageScale = renderStack.displays[0].width/(renderStack.width)
    pixelDiameter = imageScale*renderStack.displays[0].pixelSize()[0]

    gaussMip = dgm.warp.GaussMIPMap('imageGaussMip')
    renderStack.shaders['imageGaussMip'] = gaussMip
    #print(gaussMip.fragmentShader)

    dof = dgm.warp.DepthOfField('depthOfField')
    renderStack.shaders['depthOfField'] = dof

    renderStack.append(camera)
    renderStack.append(gaussMip)
    renderStack.append(dof)

    return True                                                         # Initialization Successful

def animateScene(renderStack, frame):
    # infinity rotate:
    y = math.sin(frame*math.pi/60)
    x = math.cos(frame*math.pi/30)/4
    for obj in renderStack.objects.values():
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
        print("Current focal depth = %s"%renderStack.focus)
    else:                   # print "down"
        renderStack.focus -= .1
        renderStack.focusChanged = True
        print("Current focal depth = %s"%renderStack.focus)

def drawScene(renderStack):
    ''' Draw everything in renderStack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our stack to screen

def setup():
    renderStack = ui.RenderStack()
    renderStack.displays.append(ui.Display(resolution=(1920,1200),size=(.518,.324)))
    ui.init()
    offset = (-1920,0)
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
        now = time.time()
        time.sleep(max((frame+1)/config.maxFPS+start-now,0))
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
