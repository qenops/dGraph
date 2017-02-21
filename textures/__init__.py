#!/usr/bin/env python
'''A library of texture methods and utilities for managing textures for openGL rendering

David Dunn
Jan 2017 - Created

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.0'

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import shaders
from OpenGL.GL import *
import cv2
import os
import dGraph.util.imageManip as dgim

# Get numchannels from format:
glFormatChannels={GL_STENCIL_INDEX:1, GL_DEPTH_COMPONENT:1, GL_DEPTH_STENCIL:1, GL_RED:1, GL_GREEN:1, GL_BLUE:1, GL_RGB:3, GL_BGR:3, GL_RGBA:4, GL_BGRA:4}
# Get format from numchanels:
glChannelsFormat={1:GL_LUMINANCE, 2:GL_LUMINANCE_ALPHA, 3:GL_RGB, 4:GL_RGBA}
# Get numpy dtype from gltype:
glTypeToNumpy={
    GL_UNSIGNED_BYTE:np.uint8,
    GL_BYTE:np.int8,
    GL_UNSIGNED_SHORT:np.uint16,
    GL_SHORT:np.int16,
    GL_UNSIGNED_INT:np.uint32,
    GL_INT:np.int32,
    #GL_HALF_FLOAT,
    GL_FLOAT:np.float32,
    #GL_UNSIGNED_BYTE_3_3_2,
    #GL_UNSIGNED_BYTE_2_3_3_REV,
    #GL_UNSIGNED_SHORT_5_6_5,
    #GL_UNSIGNED_SHORT_5_6_5_REV,
    #GL_UNSIGNED_SHORT_4_4_4_4,
    #GL_UNSIGNED_SHORT_4_4_4_4_REV,
    #GL_UNSIGNED_SHORT_5_5_5_1,
    #GL_UNSIGNED_SHORT_1_5_5_5_REV,
    #GL_UNSIGNED_INT_8_8_8_8,
    #GL_UNSIGNED_INT_8_8_8_8_REV,
    #GL_UNSIGNED_INT_10_10_10_2,
    #GL_UNSIGNED_INT_2_10_10_10_REV,
    #GL_UNSIGNED_INT_24_8,
    #GL_UNSIGNED_INT_10F_11F_11F_REV,
    #GL_UNSIGNED_INT_5_9_9_9_REV,
    #GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
}
# Get gltype from numpy dtype:
glNumpyToType={
    np.uint8:GL_UNSIGNED_BYTE,
    np.int8:GL_BYTE,
    np.uint16:GL_UNSIGNED_SHORT,
    np.int16:GL_SHORT,
    np.uint32:GL_UNSIGNED_INT,
    np.int32:GL_INT,
    np.float32:GL_FLOAT,
}
def prepareImage(image):
    iy, ix, channels = image.shape if len(image.shape)>2 else [image.shape[0], image.shape[1], 1]
    img = image
    #if channels == 1:
    #    img = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB),cv2.COLOR_RGB2RGBA)
    #    channels = 4
    if channels == 4:
        if image.dtype.type == np.float64:  # if they are float64, they were loaded as numpy arrays, so don't swap channels
            img = image.astype(np.float32)
        else:
            img = cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA)
    elif channels == 3:  # Nvidia cards having trouble with 3 channel images, so convert it to 4
        if image.dtype.type == np.float64:  # if they are float64, they were loaded as numpy arrays, so don't swap channels
            img = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2RGBA)
        else:
            img = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
        channels = 4
    img = np.flipud(img)
    format = glChannelsFormat[channels]
    type = glNumpyToType[img.dtype.type]
    # This is ugly (using exec) but I can't figure another way to do it
    formatBase = str(format).split()[0]
    mod = ''
    if 'float' in img.dtype.name:
        mod = 'F'
        if channels == 1:
            formatBase = 'GL_R'
    #elif 'u' in img.dtype.name:
    #    mod = 'UI'
    tempISF = '%s%s%s'%(formatBase,dgim.strToInt(img.dtype.name),mod)
    exec('internalSizedFormat = %s'%tempISF) 
    return img, iy, ix, channels, format, type, internalSizedFormat

def createTexture(image, numMipmaps=1,wrap=GL_REPEAT,filter=GL_LINEAR,mipfilter=GL_LINEAR_MIPMAP_LINEAR):
    img, height, width, channels, format, type, isf = prepareImage(image)
    print '%s, %s, %s, %s, %s, %s'%(height, width, channels, format, type, isf)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    #glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, type, img)
    glTexStorage2D(GL_TEXTURE_2D, numMipmaps, isf, width, height)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    if numMipmaps < 1:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mipfilter)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter)
    # we should do some verification if the texture is as it should be
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT)
    #int(GL_RGBA)
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_RED_TYPE)
    #int(GL_UNSIGNED_NORMALIZED)
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_RED_SIZE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture

def updateTexture(texture, image):
    img, height, width, channels, format, type, isf = prepareImage(image)
    glBindTexture(GL_TEXTURE_2D, texture)
    # we should do some verification if the texture is as it should be
    #width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    #height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
    #texFormat = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT)
    #texType = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_TYPE)
    #if ix == width and iy == height and format == texFormat and type == texType:
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, img)
    #else:
    #   pass
    glBindTexture(GL_TEXTURE_2D, 0)

def attachTexture(texture, shader, index):
    GL.glActiveTexture(getattr(GL,'GL_TEXTURE%s'%index))                    # make texture register idx active
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)                             # bind texture to register idx
    texLoc = GL.glGetUniformLocation(shader, "tex%s"%index)                 # get location of our texture
    GL.glUniform1i(texLoc, index)                                           # connect location to register idx

def createWarp(width, height, type=GL_UNSIGNED_BYTE,wrap=GL_MIRRORED_REPEAT,filter=GL_LINEAR):
    texture = glGenTextures(1)                           # setup our texture
    glBindTexture(GL_TEXTURE_2D, texture)                # bind texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, type, None)   # Allocate memory
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    # setup frame buffer
    frameBuffer = glGenFramebuffers(1)                                                                  # Create frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer)                                                   # Bind our frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)        # Attach texture to frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture, (frameBuffer,width,height)

def readFramebuffer(x, y, width, height, format, gltype=GL.GL_UNSIGNED_BYTE):
    '''Get pixel values from framebuffer to numpy array'''
    stringArry = GL.glReadPixels(x,y,width,height,format,gltype)        # read the buffer
    arry = np.fromstring(stringArry,glTypeToNumpy[gltype])              # convert back to numbers
    arry = np.reshape(arry,(height,width,glFormatChannels[format]))     # reshape our array to right dimensions
    arry = np.flipud(arry)                                              # openGL and openCV start images at bottom and top respectively, so flip it
    if glFormatChannels[format] > 2:                                    # swap red and blue channel
        temp = np.zeros_like(arry)
        np.copyto(temp, arry)
        temp[:,:,0] = arry[:,:,2]
        temp[:,:,2] = arry[:,:,0]
        arry=temp
    return arry

def loadImage(imgFile):
    ''' Load an image from a file '''
    filename, extension = os.path.splitext(imgFile)
    if extension == '.npy':
        image = np.load(imgFile)
    elif extension == '.png':
        image = cv2.imread(imgFile, -1)
    else:
        raise ValueError('File type unknown')
    return image