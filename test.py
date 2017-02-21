#!/usr/bin/python
'''A simple openGL based stereo renderer

David Dunn
July 2015 - created

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '0.1'

import sys
sys.path.append(u'../')
import OpenGL
OpenGL.ERROR_CHECKING = False      # Uncomment for 2x speed up
OpenGL.ERROR_LOGGING = False       # Uncomment for speed up
#OpenGL.FULL_LOGGING = True         # Uncomment for verbose logging
#OpenGL.ERROR_ON_COPY = True        # Comment for release
import OpenGL.GL as GL
#import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
from OpenGL.GL import shaders
import numpy as np
import dGraph as dg
from dGraph import shaders as dgs
from dGraph import textures as dgt
from dGraph import imageManip as im
import dDisplay as disp
from time import time
import cv2

modelDir = './objs'

def loadCrosses(file=None):                
    '''Load or create our sceneGraph'''
    global renderStack, cameras, objects
    scene = dg.SceneGraph(file)
    stereoCam = dg.StereoCamera('front', scene)
    stereoCam.setResolution((width/2, height))
    stereoCam.setTranslate(0.,0.,0.)
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
        cross = dg.PolySurface('cross%s'%idx, scene, file = '%s/cross.obj'%modelDir)
        cross.setScale(.01,.01,.01)
        cross.translate = position
        objects[cross.name] = cross
        print(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.)
        material = dgs.Material('material%s'%idx,ambient=(1,(idx/3.)/3.+1/3.,(idx%3)/3.+1/3.), amb_coeff=.5)
        #material = dgs.Lambert('material%s'%idx,ambient=(1,0,0), amb_coeff=.5, diffuse=(1,1,1), diff_coeff=1)
        cross.setMaterial(material)
    cameras = [stereoCam]
    renderStack.append(stereoCam)
    graphicsCardInit()
    return True 

def loadImageFlip():
    global renderStack, cameras, objects, display


def loadScene(file=None):                
    '''Load or create our sceneGraph'''
    global renderStack, cameras, objects, display

    switch = True   # crosseye rendering
    #switch = False
    IPD = .092
    #IPD = -.030
    
    backScene = dg.SceneGraph(file)
    bstereoCam = dg.StereoCamera('back', backScene, switch)
    bstereoCam.setResolution((width/2, height))
    bstereoCam.setTranslate(0.,0.,0.)
    bstereoCam.setFOV(50.)
    bstereoCam.IPD = IPD
    bstereoCam.far = 10
    cameras.append(bstereoCam)
    cube = dg.PolySurface('cube', backScene, file = '%s/cube.obj'%modelDir)
    cube.setScale(.2,.2,.2)
    cube.setTranslate(0.,0.,-10.)
    cube.setRotate(25.,65.,23.)
    objects['cube'] = cube
    #plane = dg.PolySurface('plane', backScene, file = '%s/objects/wedge.obj'%modelDir)
    #objects['plane'] = plane
    #plane.setScale(.005,.005,.005)
    #plane.setRotate(0.,180.,0.)
    #plane.setTranslate(.0,.0,-10.)
    teapot = dg.PolySurface('teapot', backScene, file = '%s/teapot.obj'%modelDir)
    teapot.setScale(.1,.1,.1)
    teapot.setTranslate(.0,-.05,-4.)
    teapot.setRotate(5.,0.,0.)
    objects['teapot'] = teapot
    
    #frontScene = dg.SceneGraph(file)
    #stereoCam = dg.StereoCamera('front', frontScene, switch)
    #stereoCam.setResolution((width/2, height))
    #stereoCam.translate.connect(bstereoCam.translate)
    #stereoCam.rotate.connect(bstereoCam.rotate)
    #stereoCam.setFOV(50.)
    #stereoCam.IPD.connect(bstereoCam.IPD)
    #sphere = dg.PolySurface.polySphere('sphere',frontScene)
    #sphere.setScale(.05,.05,.05)
    #sphere.setTranslate(-.05,0.,-6.)
    #objects['sphere'] = sphere
    
    #material1 = dgs.Lambert('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    material1 = dgs.Test('material1',ambient=(1,0,0), amb_coeff=0.2, diffuse=(1,1,1), diff_coeff=1)
    for obj in objects.itervalues():
        obj.setMaterial(material1)
    
    imageScale = display.resolution[0]/(width/2.)
    pixelDiameter = imageScale*display.pixelSize()[0]
    ftBlur = dgs.Convolution('ftBlur')
    kernel = im.getPSF(10., 8., aperture=.004, pixelDiameter=pixelDiameter)
    ftBlur.kernel = kernel
    #stereoCam.rightStackAppend(ftBlur)
    
    bkBlur = dgs.Convolution('bkBlur')
    kernel = im.getPSF(8., 10., aperture=.004, pixelDiameter=pixelDiameter)
    bkBlur.kernel = kernel
    #bstereoCam.leftStackAppend(bkBlur)

    #over = dgs.Over('over')
    #over.overStackAppend(stereoCam.left)
    #over.underStackAppend(bstereoCam.left)
    #over.overStackAppend(stereoCam)
    #over.underStackAppend(bstereoCam)
    #renderStack.append(over)
    renderStack.append(bstereoCam)

    #warp = dgs.Lookup('lookup', 'warpWorking.png')
    #renderStack.append(warp)

    graphicsCardInit()
    return True                                                         # Initialization Successful

def graphicsCardInit():
    ''' compile shaders and create VBOs and such '''
    global renderStack, width, height
    sceneGraphSet = set()
    for node in renderStack:
        sceneGraphSet.update(node.setup(width, height))
    for sceneGraph in sceneGraphSet:
        for obj in sceneGraph:                                                      # convert the renderable objects in the scene
            if obj.renderable:
                print obj.name
                obj.generateVBO()

def initGL():
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_GREATER)
    GL.glDepthRange(0,1)
    GL.glClearDepth(0)
    GL.glClearColor(0, 0, 0, 0)

    #GL.glEnable(GL.GL_TEXTURE_2D)                                      # Not needed for shaders?
    #GL.glEnable(GL.GL_NORMALIZE)                                       # Enable normal normalization

def resizeWindow(w, h):
    global renderStack, cameras, width, height
    width = w if w > 1 else 2
    height = h if h > 1 else 2
    for cam in cameras:
        cam.setResolution((width/2, height))
    for node in renderStack:
        node.setup(width, height)
    
def DrawGLScene():
    ''' Draw everything in stereo '''
    global renderStack, frameCount, cameras, width, height
    # Get Window Dimensions
    #w = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)
    #h = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
    #if w != width or h != height:
    #    width = w
    #    height = h
    #    for cam in cameras:
    #        cam.setResolution((width/2, height))
    #    # should probably update warp texture subimage size here
    myStack = list(renderStack)                                     # copy the renderStack so we can pop and do it again next frame
    #myStack = renderStack
    temp = myStack.pop()
    temp.render(width, height, myStack)                    # Render our warp to screen
    #data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)
    #data = np.reshape(data,(height,width,3))
    #cv2.imshow('%s:%s'%('final frame',temp._name),data)
    #cv2.waitKey()
    #GLUT.glutPostRedisplay()
    GLUT.glutSwapBuffers()
    
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
    endTime = time()
    frameCount+=1
    elapsed = endTime-startTime
    fps = frameCount / elapsed
    GLUT.glutSetWindowTitle("Binocular Depth Fusion: %0.2f FPS"%fps);


# The function called whenever a key is pressed. Tuples to pass in: (key, x, y)  
def keyPressed(*args):
    ''' Directory of key presses:
    Camera Motions:
        Translate:  w = up
                    s = down
                    a = left
                    d = right
        Rotate:     i = -z
                    k = +z
                    j = -x
                    l = +x
        IPD:        z = increase IPD
                    x = decrease IPD
        Output:     p = Print camera info
    Deconvolution:
        Noise to Signal Ratio:  [ = decrease nsr
                                ] = increase nsr
        Aperture size:          - = smaller
                                = = bigger
    Write Images (three buffers for each):
        Standard Blur (What you see is what you get):   [0-3]
        Deconvolution Blur:                             [4-6]
        Band Stop Filter:                               [7-9]
    '''
    global cameras, objects, nsr, aperture
    translate = np.array((0.,0.,0.))
    rotate = np.array((0.,0.,0.))
    ipd = 0
    ESCAPE = '\033'
    # If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit ()
    if args[0] == 'w':
        translate -= np.array((0.,0.,0.1))
    if args[0] == 's':
        translate += np.array((0.,0.,0.1))
    if args[0] == 'a':
        translate -= np.array((0.005,0.,0.))
    if args[0] == 'd':
        translate += np.array((0.005,0.,0.))
    if args[0] == 'i':
        rotate -= np.array((0.,5.,0.))
    if args[0] == 'k':
        rotate += np.array((0.,5.,0.))
    if args[0] == 'j':
        rotate -= np.array((5.,0.,0.))
    if args[0] == 'l':
        rotate += np.array((5.,0.,0.))
    if args[0] == 't':
        objects['teapot'].translate -= np.array((0.,0.,0.1))
    if args[0] == 'g':
        objects['teapot'].translate += np.array((0.,0.,0.1))
    if args[0] == 'z':
        ipd += 0.005
    if args[0] == 'x':
        ipd -= 0.005
    if args[0] == '[':
        nsr *= 0.75
    if args[0] == ']':
        nsr *= 4/3.
    if args[0] == '-':
        aperture -= 0.001
    if args[0] == '=':
        aperture += 0.001
    if args[0] == 'p':
        print('Translate = %s'%cameras[0].translate)
        print('Rotate = %s'%cameras[0].rotate)
        print('IPD = %s'%cameras[0].IPD)
        print('nsr = %s'%nsr)
        print('aperture = %s'%aperture)
    for cam in cameras:
        cam.IPD += ipd
    for name, obj in objects.iteritems():
        obj.translate += translate
        obj.rotate += rotate
    if isinstance(args[0], basestring) and args[0].isdigit():
        writeImages(int(args[0]))
    return

def writeImages(buffer):
    '''Write Images (three buffers for each):
        Standard Blur (What you see is what you get):   [0-3]
        Deconvolution Blur:                             [4-6]
        Band Stop Filter:                               [7-9]
    '''
    correctShift = False
    global renderStack, cameras, width, height, display, nsr, aperture
    if buffer <= 9:     # Standard Blur
        image = dgt.readFramebuffer(0,0,width,height,GL.GL_RGB)
        # split left and right eye
        left = image[:,:width/2]
        right = image[:,width/2:]
    elif buffer <= 6:   # Deconvolution Blur
        images = []
        depths = [objects['teapot'].translate[2],objects['sphere'].translate[2]]
        for camera in cameras:
            camera.stackSuspend()
            myStack = [camera]
            myStack.pop().render(width, height, myStack)
            images.append(dgt.readFramebuffer(0,0,width,height,GL.GL_RGBA,GL.GL_FLOAT))
            camera.stackResume()
        # This is going to be hacky - find a better way of getting image/focal depth
        composited = [np.zeros_like(images[0][:,width/2:,:3]),np.zeros_like(images[0][:,:width/2,:3])]
        imageScale = display.resolution[0]/(width/2.)
        pixelDiameter = imageScale*display.pixelSize()[0]
        for idx, image in enumerate(images):
            imageDepth = depths[idx]
            lf = image[:,:width/2]
            rt = image[:,width/2:]
            for dex, side in enumerate([rt,lf]):
                focusDepth = depths[dex]
                if imageDepth != focusDepth:
                    psf = im.getPSF(abs(focusDepth), abs(imageDepth), aperture=aperture, pixelDiameter=pixelDiameter)
                    topAlpha = im.deconvolveWiener(side[:,:,3],psf,nsr)[:,:,0]
                    topColor =  im.deconvolveWiener(cv2.multiply(side[:,:,:3],np.dstack((side[:,:,3],side[:,:,3],side[:,:,3]))),psf,nsr)
                else:
                    topAlpha = side[:,:,3]
                    topColor = side[:,:,:3]
                composited[dex], trash = im.over(topColor, topAlpha, composited[dex])
        scale = float(im.getBitDepthScaleFactor('uint8'))
        right = np.uint8(composited[0]*scale)
        left = np.uint8(composited[1]*scale)
    else:               # Band Stop filter
        pass
    # scale the right eye
    #right = im.scaleImgDist(10., 8., right, (right.shape[1],right.shape[0]), 1.0)
    #interp = cv2.INTER_NEAREST
    #right = cv2.resize(right, tuple([int(round(i*1.05)) for i in (left.shape[1],left.shape[0])]), 0, 0,interp)
    #right = im.cropImg(right, (left.shape[0],left.shape[1]))
    if correctShift:
        # calculate pixel shift to display correct IPD
        imageScale = display.resolution[0]/(width/2.)
        shift = int(round((display.size[0]/2. + display.bezel[0] - cameras[0].IPD/2.) / (imageScale*display.pixelSize()[0])))  
        left = np.roll(left, shift, 1)            # correct images for display ipd
        right = np.roll(right, -shift, 1)
    #buffer = 'calibration'
    cv2.imwrite('./leftRenders/output-%s.png'%buffer, left)
    cv2.imwrite('./rightRenders/output-%s.png'%buffer, right)
    print("Wrote files to output-%s.png"%buffer)
    cv2.imshow('Left output-%s.png'%buffer,left)
    cv2.waitKey()

if __name__ == '__main__':
    display = disp.Display()
    renderStack = []
    cameras = []
    objects = {}
    width, height = (1920, 1080)
    #width, height = (1080, 1080)
    nsr, aperture = 25, .001  # noise to signal for deconvolution
    winName = 'Binocular Depth Fusion'
    # Print message to console, and kick off the main to get it rolling.
    print("Hit ESC key to quit.")
    # pass arguments to init
    GLUT.glutInit(sys.argv)
    #GLUT.glutInitDisplayMode(GLUT.GLUT_RGB | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
    GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
    GLUT.glutInitWindowSize(width, height)
    # the window starts at the upper left corner of the screen 
    GLUT.glutInitWindowPosition(0, 0)
    window = GLUT.glutCreateWindow(winName)
    GLUT.glutDisplayFunc(DrawGLScene)
    # Uncomment this line to get full screen.
    #glutFullScreen()
    # When we are doing nothing, redraw the scene.
    GLUT.glutIdleFunc(DrawGLScene)
    GLUT.glutReshapeFunc(resizeWindow)
    GLUT.glutKeyboardFunc(keyPressed)
    GLUT.glutSpecialFunc(keyPressed)
    loadScene()
    #loadCrosses()
    initGL()
    # Timing code
    frameCount = 0
    startTime = time()
    # do a raster by hand verification
    #frame, depth = cameras[0].raster(1)
    #cv2.namedWindow(winName)
    #cv2.imshow(winName, frame)
    #cv2.imshow('%s Depth'%winName, depth)
    #cv2.waitKey()
    # Start Event Processing Engine 
    GLUT.glutMainLoop()
