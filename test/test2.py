#!/usr/bin/python
'''Test for an openGL based stereo renderer - test distortion warp texture

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
import cv2
import numpy as np
import dDisplay as dd
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
    teapot.setScale(.1,.1,.1)
    teapot.setTranslate(.0,-.05,-2.)
    teapot.setRotate(5.,0.,0.)
    renderStack.objects['teapot'] = teapot

    material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in renderStack.objects.itervalues():
        obj.setMaterial(material1)

    renderStack.append(cam)
    warp = dgm.warp.Lookup('lookup1',lutFile='%s/warp_0000.npy'%MODELDIR)
    renderStack.append(warp)

def arrowKey(window,renderStack,direction):
    if direction == 3:    # print "right"
        renderStack.objects['teapot'].rotate += np.array((0.,5.,0.))
    elif direction == 2:    # print "left"
        renderStack.objects['teapot'].rotate -= np.array((0.,5.,0.))
    elif direction == 1:      # print 'up'
        renderStack.objects['teapot'].translate += np.array((0.,.01,0.))
    else:                   # print "down"
        renderStack.objects['teapot'].translate -= np.array((0.,.01,0.))

def drawGLScene(renderStack):
    ''' Render the stack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our warp to screen

def runTest():
    renderStack = ui.RenderStack()
    renderStack.display = dd.Display()
    ui.init()
    offset = (1920,0)
    mainWindow = renderStack.addWindow(ui.open_window('Warp Distortion Test', offset[0], offset[1], renderStack.display.width, renderStack.display.height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    x, y = ui.get_window_pos(mainWindow)
    width, height = ui.get_window_size(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=renderStack, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=renderStack, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=renderStack, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=renderStack, direction=0)
    dg.initGL()
    loadScene(renderStack)
    renderStack.graphicsCardInit()
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        drawGLScene(renderStack)
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        #ui.wait_events()
    ui.terminate()
    exit(0)

if __name__ == '__main__':
    runTest()