#!/usr/bin/python
'''Test for an openGL based stereo renderer

David Dunn
July 2015 - created
Jan 2017 - transition to GLFW

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.0'

import dglfw as fw
import OpenGL
OpenGL.ERROR_CHECKING = False      # Uncomment for 2x speed up
OpenGL.ERROR_LOGGING = False       # Uncomment for speed up
#OpenGL.FULL_LOGGING = True         # Uncomment for verbose logging
#OpenGL.ERROR_ON_COPY = True        # Comment for release
import OpenGL.GL as GL
import cv2
import numpy as np
import dGraph as dg
import dDisplay as dd

modelDir = './data'

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
    dg.graphicsCardInit(renderStack, width, height)
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

    dg.graphicsCardInit(renderStack, width, height)
    return True                                                         # Initialization Successful

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
    
def DrawGLScene():
    ''' Draw everything in stereo '''
    global renderStack, frameCount, cameras, width, height
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
    #endTime = time()
    #frameCount+=1
    #elapsed = endTime-startTime
    #fps = frameCount / elapsed
    

def runTest():
    display = dd.Display()
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
    fw.init()
    loadScene()
    #loadCrosses()
    dg.initGL()
    # Timing code
    #frameCount = 0
    #startTime = time()
    # do a raster by hand verification
    #frame, depth = cameras[0].raster(1)
    #cv2.namedWindow(winName)
    #cv2.imshow(winName, frame)
    #cv2.imshow('%s Depth'%winName, depth)
    #cv2.waitKey()


if __name__ == '__main__':
    runTest()