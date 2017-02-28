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
import math
import numpy as np
import dGraph as dg
import dGraph.ui as ui
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.materials as dgm
import dGraph.materials.warp
import dGraph.util.imageManip as im

MODELDIR = './dGraph/test/data'

def loadScene(renderStack,file=None):                
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph(file)
    cam = dgc.Camera('cam', scene)
    cam.setResolution((renderStack.width, renderStack.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    renderStack.cameras.append(cam)
    teapot = dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR)
    teapot.setScale(.4,.4,.4)
    teapot.setTranslate(0.,-.20,-1.)
    teapot.setRotate(0.,0.,0.)
    renderStack.objects['teapot'] = teapot

    material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in renderStack.objects.values():
        obj.setMaterial(material1)

    renderStack.append(cam)
    #warp = dgm.warp.Lookup('lookup1',lutFile='%s/warp_0020.npy'%MODELDIR)
    #renderStack.append(warp)
    return scene

def animateScene(renderStack, frame):
    ''' Create motion in our scene '''
    # infinity rotate:
    y = 1
    x = math.cos(frame*math.pi/60)
    for obj in renderStack.objects.values():
        obj.rotate += np.array((x,y,0.))
    
def drawGLScene(renderStack):
    ''' Draw everything in renderStack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our camera to screen

def addInput(renderStack):
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=renderStack, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=renderStack, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=renderStack, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=renderStack, direction=0)

''' In order to do input with mouse and keyboard, we need to setup a state machine with 
states switches from mouse presses and releases and certain key presses 
- maybe enable a glfwSetCursorPosCallback when button is pressed, or
- more likely just poll cursor position since you can't disable a callback'''
def arrowKey(window,renderStack,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        renderStack.objects['teapot'].rotate += np.array((5.,0.,0.))
        print(renderStack.objects['teapot'].rotate)
    else:                   # print "down"
        renderStack.objects['teapot'].rotate -= np.array((5.,0.,0.))
        print(renderStack.objects['teapot'].rotate)

def drawScene(renderStack):
    ''' Render the stack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our warp to screen

def setup():
    renderStack = ui.RenderStack()
    renderStack.displays.append(ui.Display(resolution=(1920,1200)))
    ui.init()
    offset = (0,0)
    mainWindow = renderStack.addWindow(ui.open_window('Scene Graph Test', offset[0], offset[1], renderStack.displays[0].width, renderStack.displays[0].height))
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
    print("Use Up/Down to rotate the solar system.")
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
