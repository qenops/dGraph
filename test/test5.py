#!/usr/bin/python
'''Test for an openGL based stereo renderer - test binocular rendering to a single window

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
WINDOWS = [{
    "name": 'Test 5',
    #"location": (0, 0),
    "location": (2436, 1936), # px coordinates of the startup screen for window location
    #"size": (1920, 1080),
    "size": (1600,800), # px size of the startup screen for centering
    "center": (400,400), # center of the display
    "refresh_rate": 60, # refreshrate of the display for precise time measuring
    "px_size_mm": 0.09766, # px size of the display in mm
    "distance_cm": 20, # distance from the viewer in cm,
    #"is_hmd": False,
    #"warp_path": 'data/calibration/newRight/',
    },
]

def loadScene(renderStack, file=None, cross=False):                
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph(file)
    stereoCam = dgc.StereoCamera('front', scene)
    stereoCam.setResolution((renderStack.width/2, renderStack.height))
    stereoCam.setTranslate(0.,0.,0.)
    stereoCam.setFOV(50.)
    stereoCam.IPD = .062
    stereoCam.midOffset = -52
    if cross:
        crosses = [
            np.array((-.4,-.4,-2.)),
            np.array((-.4,.0,-2.)),
            np.array((-.4,.4,-2.)),
            np.array((.0,-.4,-2.)),
            np.array((.0,.0,-2.)),
            np.array((.0,.4,-2.)),
            np.array((.4,-.4,-2.)),
            np.array((.4,.0,-2.)),
            np.array((.4,.4,-2.)),
        ]
        for idx, position in enumerate(crosses):
            cross = dgs.PolySurface('cross%s'%idx, scene, file = '%s/cross.obj'%MODELDIR)
            cross.setScale(.02,.02,.02)
            cross.translate = position
            renderStack.objects[cross.name] = cross
            #print(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.)
            material = dgm.Material('material%s'%idx,ambient=(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.), amb_coeff=.5)
            #material = dgm.Lambert('material%s'%idx,ambient=(1,0,0), amb_coeff=.5, diffuse=(1,1,1), diff_coeff=1)
            cross.setMaterial(material)
    else:
        teapot = dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR)
        teapot.setTranslate(0.,0.,-.2)
        teapot.setScale(.05,.05,.05)
        teapot.setRotate(0.,0.,0.)
        renderStack.objects['teapot'] = teapot
        material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
        teapot.setMaterial(material1)
    renderStack.cameras = [stereoCam]
    renderStack.append(stereoCam)
    return True 

def animateScene(renderStack, frame):
    ''' Create motion in our scene '''
    # infinity rotate:
    y = 1
    x = math.cos(frame*math.pi/60)
    for obj in renderStack.objects.itervalues():
        obj.rotate += np.array((x,y,0.))

def addInput():
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=renderStack, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=renderStack, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=renderStack, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=renderStack, direction=0)

def arrowKey(window,renderStack,direction):
    if direction == 3:    # print "right"
        renderStack.cameras[0].ipd += .001
    elif direction == 2:    # print "left"
        renderStack.cameras[0].ipd -= .001
    elif direction == 1:      # print 'up'
        renderStack.cameras[0].midOffset += 1
    else:                   # print "down"
        renderStack.cameras[0].midOffset -= 1 
        print renderStack.cameras[0].midOffset

def drawScene(renderStack):
    ''' Render the stack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our warp to screen

def setup():
    winData = WINDOWS[0]
    renderStack = ui.RenderStack()
    renderStack.display = ui.Display(resolution=winData['size'])
    ui.init()
    mainWindow = renderStack.addWindow(ui.open_window(winData['name'], winData['location'][0], winData['location'][1], renderStack.display.width, renderStack.display.height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    ui.make_context_current(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    dg.initGL()
    scene = loadScene(renderStack)
    renderStack.graphicsCardInit()
    return renderStack, scene, [mainWindow]

def runLoop(renderStack, mainWindow):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    frame = 0
    while not ui.window_should_close(mainWindow):
        ui.make_context_current(mainWindow)
        drawScene(renderStack)
        ui.swap_buffers(mainWindow)
        ui.poll_events()
        #animateScene(renderStack, frame)
        frame += 1
    ui.terminate()
    exit(0)

if __name__ == '__main__':
    renderStack, scene, windows = setup()
    addInput()
    runLoop(renderStack, windows[0])
