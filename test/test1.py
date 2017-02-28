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
    sun = dgs.PolySurface.polySphere('sun', scene, radius=.2)
    sun.setTranslate(0.,0.,-2.)
    sun.setRotate(10.,0.,0.)
    renderStack.objects['sun'] = sun
    sunMaterial = dgm.Lambert('sunMaterial',ambient=(1,1,0), amb_coeff=1., diffuse=(0,0,0), diff_coeff=0)
    sun.setMaterial(sunMaterial)
    mercury = dgs.PolySurface.polySphere('mercury', sun, radius=.01)
    mercury.setTranslate(0.3,0.,0.)
    renderStack.objects['mercury'] = mercury
    mercuryMaterial = dgm.Lambert('mercuryMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(1.,.5,.3), diff_coeff=1)
    mercury.setMaterial(mercuryMaterial)
    venus = dgs.PolySurface.polySphere('venus', sun, radius=.028)
    venus.setTranslate(0.425,0.,0.)
    renderStack.objects['venus'] = venus
    venusMaterial = dgm.Lambert('venusMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(1.,.6,.6), diff_coeff=1)
    venus.setMaterial(venusMaterial)
    earth = dgs.PolySurface.polySphere('earth', sun, radius=.03)
    earth.setTranslate(0.6,0.,0.)
    earth.setRotate(0.,0.,15.)
    renderStack.objects['earth'] = earth
    earthMaterial = dgm.Lambert('earthMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.25,.41,.879), diff_coeff=1)
    earth.setMaterial(earthMaterial)
    moon = dgs.PolySurface.polySphere('moon', earth, radius=.005)
    moon.setTranslate(0.075,0.,0.)
    renderStack.objects['moon'] = moon
    moonMaterial = dgm.Lambert('moonMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.7,.7,.7), diff_coeff=1)
    moon.setMaterial(moonMaterial)
    mars = dgs.PolySurface.polySphere('mars', sun, radius=.015)
    mars.setTranslate(0.8,0.,0.)
    renderStack.objects['mars'] = mars
    marsMaterial = dgm.Lambert('marsMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.8,.2,.2), diff_coeff=1)
    mars.setMaterial(marsMaterial)
    jupiter = dgs.PolySurface.polySphere('jupiter', sun, radius=.1)
    jupiter.setTranslate(1.2,0.,0.)
    renderStack.objects['jupiter'] = jupiter
    jupiterMaterial = dgm.Lambert('jupiterMaterial',ambient=(0,0,0), amb_coeff=0., diffuse=(.867,.71875,.527), diff_coeff=1)
    jupiter.setMaterial(jupiterMaterial)
    renderStack.append(cam)
    return True                                                         # Initialization Successful

def animateScene(renderStack, frame):
    ''' Create motion in our scene '''
    #          name:    (distance,  orbit,     rotation )
    speeds = {'mercury':(0.3,       87.969,     58.646  ),
              'venus':  (0.425,    224.7,      243      ),
              'earth':  (0.6,      365.2564,     1      ),
              'moon':   (0.075,     28,         28      ),
              'mars':   (0.8,      687,          1.03   ),
              'jupiter':(1.2,     4332.59,       0.4135 )} 
    for k, v in renderStack.objects.items():
        if k in speeds:
            dist, orbit, rot = speeds[k]
            # add rot when we have textures -  but will mess up children
            x =  math.cos(math.pi/2/orbit*frame) * dist
            z =  math.sin(math.pi/2/orbit*frame) * dist
            v.translate = np.array((x,0.,z))
    
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
        renderStack.objects['sun'].rotate += np.array((5.,0.,0.))
        print(renderStack.objects['sun'].rotate)
    else:                   # print "down"
        renderStack.objects['sun'].rotate -= np.array((5.,0.,0.))
        print(renderStack.objects['sun'].rotate)

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
