#!/usr/bin/python
'''Test for an openGL based renderer - testing scene graph features (parenting, dirty propigation, animation)

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
import math, os
import numpy as np
import dGraph as dg
import dGraph.ui as dgui
import dGraph.render as dgr
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.materials as dgm
import dGraph.config as config
import dGraph.util.imageManip as im
import time

MODELDIR = '%s/data'%os.path.dirname(__file__)

def loadScene(renderGraph):
    '''Load or create our sceneGraph'''
    scene = dg.SceneGraph('Test1_SG')
    cam = scene.add(dgc.Camera('cam', scene))
    cam.setResolution((renderGraph.width, renderGraph.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    sun = scene.add(dgs.PolySurface.polySphere('sun', scene, radius=.2))
    sun.setTranslate(0.,0.,-2.)
    sun.setRotate(10.,0.,0.)
    sunMaterial = scene.add(dgm.Lambert('sunMaterial',ambient=(1,1,0), amb_coeff=1., diffuse=(0,0,0), diff_coeff=0))
    sun.setMaterial(sunMaterial)
    mercury = scene.add(dgs.PolySurface.polySphere('mercury', sun, radius=.01))
    mercury.setTranslate(0.3,0.,0.)
    mercuryMaterial = scene.add(dgm.Lambert('mercuryMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(1.,.5,.3), diff_coeff=1))
    mercury.setMaterial(mercuryMaterial)
    venus = scene.add(dgs.PolySurface.polySphere('venus', sun, radius=.028))
    venus.setTranslate(0.425,0.,0.)
    venusMaterial = scene.add(dgm.Lambert('venusMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(1.,.6,.6), diff_coeff=1))
    venus.setMaterial(venusMaterial)
    earth = scene.add(dgs.PolySurface.polySphere('earth', sun, radius=.03))
    earth.setTranslate(0.6,0.,0.)
    earth.setRotate(0.,0.,15.)
    earthMaterial = scene.add(dgm.Lambert('earthMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.25,.41,.879), diff_coeff=1))
    earth.setMaterial(earthMaterial)
    moon = scene.add(dgs.PolySurface.polySphere('moon', earth, radius=.005))
    moon.setTranslate(0.075,0.,0.)
    moonMaterial = scene.add(dgm.Lambert('moonMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.7,.7,.7), diff_coeff=1))
    moon.setMaterial(moonMaterial)
    mars = scene.add(dgs.PolySurface.polySphere('mars', sun, radius=.015))
    mars.setTranslate(0.8,0.,0.)
    marsMaterial = scene.add(dgm.Lambert('marsMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.8,.2,.2), diff_coeff=1))
    mars.setMaterial(marsMaterial)
    jupiter = scene.add(dgs.PolySurface.polySphere('jupiter', sun, radius=.1))
    jupiter.setTranslate(1.2,0.,0.)
    jupiterMaterial = scene.add(dgm.Lambert('jupiterMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.867,.71875,.527), diff_coeff=1))
    jupiter.setMaterial(jupiterMaterial)
    renderGraph.frameBuffer.connectInput(cam)
    scene.add(renderGraph)
    return scene                                                         # Initialization Successful

def animateScene(scene, frame):
    ''' Create motion in our scene '''
    #          name:    (distance,  orbit,     rotation )
    speeds = {'mercury':(0.3,       87.969,     58.646  ),
              'venus':  (0.425,    224.7,      243      ),
              'earth':  (0.6,      365.2564,     1      ),
              'moon':   (0.075,     28,         28      ),
              'mars':   (0.8,      687,          1.03   ),
              'jupiter':(1.2,     4332.59,       0.4135 )} 
    for k in scene.shapes:
        if k in speeds:
            dist, orbit, rot = speeds[k]
            # add rot when we have textures -  but will mess up children
            x =  math.cos(math.pi/2/orbit*frame) * dist
            z =  math.sin(math.pi/2/orbit*frame) * dist
            scene[k].translate = np.array((x,0.,z))

def addInput(scene):
    dgui.add_key_callback(arrowKey, dgui.KEY_RIGHT, scene=scene, direction=3)
    dgui.add_key_callback(arrowKey, dgui.KEY_LEFT, scene=scene, direction=2)
    dgui.add_key_callback(arrowKey, dgui.KEY_UP, scene=scene, direction=1)
    dgui.add_key_callback(arrowKey, dgui.KEY_DOWN, scene=scene, direction=0)

''' In order to do input with mouse and keyboard, we need to setup a state machine with 
states switches from mouse presses and releases and certain key presses 
- maybe enable a glfwSetCursorPosCallback when button is pressed, or
- more likely just poll cursor position since you can't disable a callback'''
def arrowKey(window,scene,direction):
    if direction == 3:    # print "right"
        pass
    elif direction == 2:    # print "left"
        pass
    elif direction == 1:      # print 'up'
        scene['sun'].rotate += np.array((5.,0.,0.))
        print(scene['sun'].rotate)
    else:                   # print "down"
        scene['sun'].rotate -= np.array((5.,0.,0.))
        print(scene['sun'].rotate)

def setup():
    renderGraph = dgr.RenderGraph('Test1_RG')
    display = renderGraph.add(dgui.Display('Fake Display',resolution=(1920,1200)))
    dgui.init()
    offset = (0,0)
    mainWindow = renderGraph.add(dgui.open_window('Scene Graph Test', offset[0], offset[1], display.width, display.height))
    if not mainWindow:
        dgui.terminate()
        exit(1)
    x, y = dgui.get_window_pos(mainWindow)
    width, height = dgui.get_window_size(mainWindow)
    dgui.add_key_callback(dgui.close_window, dgui.KEY_ESCAPE)
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
    while not dgui.window_should_close(mainWindow):
        dgui.make_context_current(mainWindow)
        scene.render()
        now = time.time()
        toSleep = max(0,(frame+1)/config.maxFPS+start-now)
        time.sleep(toSleep)
        dgui.swap_buffers(mainWindow)
        dgui.poll_events()
        animateScene(scene, frame)
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
    scene, windows = setup()
    addInput(scene)
    runLoop(scene, windows[0])
