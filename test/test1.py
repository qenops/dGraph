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
import cv2
import numpy as np
import dGraph as dg
import dGraph.ui as ui
import dGraph.cameras as dgc
import dGraph.shapes as dgs
import dGraph.shapes.implicits as dgsi
import dGraph.materials as dgm
import dGraph.materials.warp
import dGraph.util.imageManip as im

modelDir = './dGraph/test/data'

def loadScene(renderStack,file=None):                
    '''Load or create our sceneGraph'''
    
    scene = dg.SceneGraph(file)
    cam = dgc.Camera('cam', scene)
    cam.setResolution((renderStack.width, renderStack.height))
    cam.setTranslate(0.,0.,0.)
    cam.setFOV(50.)
    renderStack.cameras.append(cam)
    sun = dgsi.Sphere('sun', scene, radius=4)
    sun.setTranslate(0.,0.,-10.)
    renderStack.objects['sun'] = sun
    sunMaterial = dgm.Lambert('sunMaterial',ambient=(1,1,0), amb_coeff=1., diffuse=(0,0,0), diff_coeff=0)
    sun.setMaterial(sunMaterial)

    #material1 = dgm.Lambert('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    #material1 = dgm.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    #for obj in renderStack.objects.itervalues():
    #    obj.setMaterial(material1)
    
    renderStack.append(cam)

    return True                                                         # Initialization Successful
    
def drawGLScene(renderStack):
    ''' Draw everything in stereo '''
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    #myStack = renderStack
    temp = myStack.pop()
    temp.render(renderStack.width, renderStack.height, myStack)                    # Render our warp to screen
    #data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)
    #data = np.reshape(data,(height,width,3))
    #cv2.imshow('%s:%s'%('final frame',temp._name),data)
    #cv2.waitKey()

    
    #for cam in cameras:
    #    cam.far = frameCount % 1000
    '''
    for obj in objects.itervalues():
        obj.rotate += (0.,2.,0.)
    
    # Write out our animation
    image = dgt.readFramebuffer(0,0,width,height,GL.GL_RGB)
    # split left and right eye
    left = image[:,:width/2]
    right = image[:,width/2:]
    cv2.imwrite('./Renders/%04d_left.png'%frameCount, left)
    cv2.imwrite('./Renders/%04d_right.png'%frameCount, right)
    if frameCount >= 180:
        sys.exit ()
    '''
    # Timing code
    #endTime = time()
    #frameCount+=1
    #elapsed = endTime-startTime
    #fps = frameCount / elapsed

''' In order to do input with mouse and keyboard, we need to setup a state machine with 
states switches from mouse presses and releases and certain key presses 
- maybe enable a glfwSetCursorPosCallback when button is pressed, or
- more likely just poll cursor position since you can't disable a callback'''
def arrowKey(window,direction):
    if direction == 3:
        print "up"
    elif direction == 2:
        print "right"
    elif direction == 1:
        print "left"
    else:
        print "down"

def runTest():
    renderStack = ui.RenderStack()
    renderStack.display = ui.Display()
    nsr, aperture = 25, .001  # noise to signal for deconvolution
    winName = 'Binocular Depth Fusion'
    # Print message to console, and kick off the main to get it rolling.
    print("Hit ESC key to quit.")
    # pass arguments to init
    ui.init()
    #
    # Timing code
    #frameCount = 0
    #startTime = time()
    # do a raster by hand verification
    #frame, depth = cameras[0].raster(1)
    #cv2.namedWindow(winName)
    #cv2.imshow(winName, frame)
    #cv2.imshow('%s Depth'%winName, depth)
    #cv2.waitKey()
    offset = (1920,0)
    mainWindow = renderStack.addWindow(ui.open_window("OpenGL_noMipMap", offset[0], offset[1], renderStack.display.width, renderStack.display.height))
    if not mainWindow:
        ui.terminate()
        exit(1)
    x, y = ui.get_window_pos(mainWindow)
    width, height = ui.get_window_size(mainWindow)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    ui.add_key_callback(arrowKey, ui.KEY_UP, direction=3)
    ui.add_key_callback(arrowKey, ui.KEY_RIGHT, direction=2)
    ui.add_key_callback(arrowKey, ui.KEY_LEFT, direction=1)
    ui.add_key_callback(arrowKey, ui.KEY_DOWN, direction=0)
    ui.make_context_current(mainWindow)
    dg.initGL()

    loadScene(renderStack)
    #loadCrosses(renderStack)
    renderStack.graphicsCardInit()
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