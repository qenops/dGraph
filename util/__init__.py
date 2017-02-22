#!/usr/bin/env python
'''Mostly a temp storage location for things I need to figure out more permanent homes for

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = []

from OpenGL.GL import *
import dGraph.ui as ui

def draw(corners=[0.,0.,1.,1.]):
    glBegin(GL_QUADS)
    glTexCoord2f(0., 0.)
    glVertex2f(corners[0], corners[1])
    glTexCoord2f(1., 0.)
    glVertex2f(corners[2], corners[1])
    glTexCoord2f(1., 1.)
    glVertex2f(corners[2], corners[3])
    glTexCoord2f(0., 1.)
    glVertex2f(corners[0], corners[3])
    glEnd()

def drawQuad(texture,corners=[0.,0.,1.,1.],size=None):
    width, height = ui.get_framebuffer_size(ui.get_current_context()) if size is None else size
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0., 1., 0., 1., 0., 1.)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    draw(corners)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

def drawWarp(frameTex, frameBuffer, lutTex, imgTex, shader):
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer[0])              # Bind our frame buffer
    drawQuad(imgTex,size=frameBuffer[1:])
    glBindFramebuffer(GL_FRAMEBUFFER, 0)                        # Disable our frameBuffer so we can render to screen
    #glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture
    #glGenerateMipmap(GL_TEXTURE_2D)                             # generate mipmap
    #glBindTexture(GL_TEXTURE_2D, 0)                             # unbind texture
    width, height = ui.get_framebuffer_size(ui.get_current_context())
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glActiveTexture(GL_TEXTURE1)                            # start frameTex: make texture register idx active
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex1")               # get location of our texture
    glUniform1i(texLoc, 1)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE0)                            # start LUTtex: make texture register idx active
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, lutTex)                        # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex0")               # get location of our texture
    glUniform1i(texLoc, 0)                                      # connect location to register idx
    draw()                                                      # draw a quad
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glUseProgram(0)

def drawMixWarp(frameTex, frameBuffer, lut1Tex, lut2Tex, factor, imgTex, shader):
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer[0])           # Bind our frame buffer
    drawQuad(imgTex,size=frameBuffer[1:])
    glBindFramebuffer(GL_FRAMEBUFFER, 0)                        # Disable our frameBuffer so we can render to screen
    width, height = ui.get_framebuffer_size(ui.get_current_context())
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glActiveTexture(GL_TEXTURE0)                            # start frameTex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex")                # get location of our texture
    glUniform1i(texLoc, 0)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE1)                            # start LUT1tex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, lut1Tex)                       # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "lut1")               # get location of our texture
    glUniform1i(texLoc, 1)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE2)                            # start LUT2tex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, lut2Tex)                       # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "lut2")               # get location of our texture
    glUniform1i(texLoc, 2)                                      # connect location to register idx
    factorLoc = glGetUniformLocation(shader, "factor")      # get the factor location
    glUniform1f(factorLoc,factor)                               # set the factor
    draw()                                                      # draw a quad
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glUseProgram(0)

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