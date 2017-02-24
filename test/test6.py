#!/usr/bin/python
'''Test for an openGL based stereo renderer - test binocular rendering to multiple windows with separate distortion warp textures

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
WINDOWS = [{
    "name": 'HMD Right',
    #"location": (0, 0),
    "location": (2436, 1936), # px coordinates of the startup screen for window location
    "size": (830, 800), # px size of the startup screen for centering
    "center": (290,216), # center of the display
    "refresh_rate": 60, # refreshrate of the display for precise time measuring
    "px_size_mm": 0.09766, # px size of the display in mm
    "distance_cm": 20, # distance from the viewer in cm,
    #"is_hmd": False,
    #"warp_path": 'data/calibration/newRight/',
    },
    {
    "name": 'HMD Left',
    #"location": (0, 0),
    "location": (3266, 1936), # px coordinates of the startup screen for window location
    "size": (830, 800), # px size of the startup screen for centering
    "center": (290,216), # center of the display
    "refresh_rate": 60, # refreshrate of the display for precise time measuring
    "px_size_mm": 0.09766, # px size of the display in mm
    "distance_cm": 20, # distance from the viewer in cm,
    #"is_hmd": False,
    #"warp_path": 'data/calibration/newRight/',
    },
]

def loadScene(renderStacks,file=None):
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph(file)
    cam = dgc.StereoCamera('cam', scene)
    cam.right.setResolution((renderStacks[0].width, renderStacks[0].height))
    cam.left.setResolution((renderStacks[1].width, renderStacks[1].height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    for rs in renderStacks:
    	rs.cameras.append(cam)
    teapot = dgs.PolySurface('teapot', scene, file = '%s/teapot.obj'%MODELDIR)
    teapot.setScale(.1,.1,.1)
    teapot.setTranslate(.0,-.05,-2.)
    teapot.setRotate(5.,0.,0.)
    for rs in renderStacks:
        rs.objects['teapot'] = teapot

    material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    teapot.setMaterial(material1)
    #for obj in renderStack.objects.itervalues():
    #    obj.setMaterial(material1)

    renderStacks[0].append(cam.right)
    warp = dgm.warp.Lookup('lookup1',lutFile='%s/warp_0020.npy'%MODELDIR)
    renderStacks[0].append(warp)

    renderStacks[1].append(cam.left)
    warp = dgm.warp.Lookup('lookup1',lutFile='%s/warp_0000.npy'%MODELDIR)
    renderStacks[1].append(warp)
    return scene

def addInput():
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, renderStack=renderStack, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, renderStack=renderStack, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_UP, renderStack=renderStack, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, renderStack=renderStack, direction=0)

def arrowKey(window,renderStack,direction):
    if direction == 3:    # print "right"
        renderStack.objects['teapot'].rotate += np.array((0.,5.,0.))
    elif direction == 2:    # print "left"
        renderStack.objects['teapot'].rotate -= np.array((0.,5.,0.))
    elif direction == 1:      # print 'up'
        renderStack.objects['teapot'].translate += np.array((0.,.01,0.))
    else:                   # print "down"
        renderStack.objects['teapot'].translate -= np.array((0.,.01,0.))

def drawScene(renderStack):
    ''' Render the stack '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)     # Render our warp to screen

def setup():
    ui.init()
    renderStacks = []
    windows = []
    for idx, winData in enumerate(WINDOWS):
        renderStack = ui.RenderStack()
        renderStack.display = ui.Display(resolution=winData['size'])
        share = None if idx == 0 else windows[0]
        window = renderStack.addWindow(ui.open_window(winData['name'], winData['location'][0], winData['location'][1], renderStack.display.width, renderStack.display.height, share=share))
        if not window:
            ui.terminate()
            exit(1)
        ui.make_context_current(window)
        dg.initGL()
        windows.append(window)
        renderStacks.append(renderStack)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    scene = loadScene(renderStacks)
    for rs in renderStacks:
    	rs.graphicsCardInit()
    return renderStacks, scene, windows

def runLoop(renderStacks, windows):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    while not ui.window_should_close(windows[0]):
        for rs in renderStacks:
            for window in rs.windows:
                ui.make_context_current(window)
                drawScene(rs)
                ui.swap_buffers(window)
        ui.poll_events()
        #ui.wait_events()
    ui.terminate()
    exit(0)

if __name__ == '__main__':
    renderStack, scene, windows = setup()
    addInput()
    runLoop(renderStack, windows)
