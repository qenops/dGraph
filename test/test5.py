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
    "location": (0, 0),
    #"location": (2436, 1936), # px coordinates of the startup screen for window location
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
    stereoCam.setTranslate(0.,-.06,0.)
    stereoCam.setRotate(20.,0.,0.)
    stereoCam.setFOV(50.)
    stereoCam.IPD = .062
    crosses = [
        #np.array((.031,.0,-10.)),
        #np.array((-.031,.0,-10.)),
        np.array((-.2,-.2,-10.)),
        np.array((-.2,.0,-10.)),
        np.array((-.2,.2,-10.)),
        np.array((.0,-.2,-10.)),
        np.array((.0,.0,-10.)),
        np.array((.0,.2,-10.)),
        np.array((.2,-.2,-10.)),
        np.array((.2,.0,-10.)),
        np.array((.2,.2,-10.)),
    ]
    for idx, position in enumerate(crosses):
        cross = dgs.PolySurface('cross%s'%idx, scene, file = '%s/cross.obj'%MODELDIR)
        cross.setScale(.01,.01,.01)
        cross.translate = position
        renderStack.objects[cross.name] = cross
        print(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.)
        material = dgm.Material('material%s'%idx,ambient=(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.), amb_coeff=.5)
        #material = dgm.Lambert('material%s'%idx,ambient=(1,0,0), amb_coeff=.5, diffuse=(1,1,1), diff_coeff=1)
        cross.setMaterial(material)
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
    for rs in renderStack:
        ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=rs, direction=3)
        ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=rs, direction=2)
        ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=rs, direction=1)
        ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=rs, direction=0)

def arrowKey(window,renderStack,direction):
    for o in renderStack.objects:
        if direction == 3:    # print "right"
            o.rotate(np.array((0.,5.,0.)))
        elif direction == 2:    # print "left"
            o.rotate(-np.array((0.,5.,0.)))
        elif direction == 1:      # print 'up'
            o.translate(np.array((0.,.01,0.)))
        else:                   # print "down"
            o.translate(-np.array((0.,.01,0.)))

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
    scenes = [loadScene(renderStack) for renderStack in renderStacks]
    for rs in renderStacks:
    	rs.graphicsCardInit()
    return renderStacks, scenes, windows

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
