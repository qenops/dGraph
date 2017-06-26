#!/usr/bin/env python
'''A library of texture methods and utilities for managing textures for openGL rendering

David Dunn
Jan 2017 - Created

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.0'

import OpenGL.GL as GL
from OpenGL.GL import *
import numpy as np
import os, math
import dGraph as dg
import dGraph.util.imageManip as dgim
import cv2

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
    ''' Take an image and convert it to the right format for loading into a texture
    Returns all the information needed for creating a texture '''
    iy, ix, channels = image.shape if len(image.shape) > 2 else [image.shape[0], image.shape[1], 1]
    img = image
    if channels == 4:
        if image.dtype.type == np.float64:  # if they are float64, they were loaded as numpy arrays, so don't swap channels
            img = image.astype(np.float32)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif channels == 3:  # Nvidia cards having trouble with 3 channel images, so convert it to 4
        if image.dtype.type == np.float64:  # if they are float64, they were loaded as numpy arrays, so don't swap channels
            img = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2RGBA)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        channels = 4
    img = np.flipud(img)
    format = glChannelsFormat[channels]
    gltype = glNumpyToType[img.dtype.type]
    # This is ugly (using exec) but I can't figure another way to do it
    #   dd - why didn't I just use setattr() ???
    formatBase = str(format).split()[0]
    mod = ''
    if 'float' in img.dtype.name:
        mod = 'F'
        if channels == 1:
            formatBase = 'GL_R'
    #elif 'u' in img.dtype.name:
    #    mod = 'UI'
    tempISF = '%s%s%s'%(formatBase,dgim.strToInt(img.dtype.name),mod)
    namespace = {}
    exec('internalSizedFormat = %s'% tempISF, None, namespace) 
    internalSizedFormat = namespace['internalSizedFormat']
    return img, iy, ix, channels, format, gltype, internalSizedFormat

def createTexture(image, mipLevels=1,wrap=GL_REPEAT,filterMag=GL_LINEAR,filterMin=GL_LINEAR_MIPMAP_LINEAR):
    ''' allocate space on the gpu and transfer the data there '''
    img, height, width, channels, format, gltype, isf = prepareImage(image)
    #print('%s, %s, %s, %s, %s, %s'%(height, width, channels, format, gltype, isf))
    if filterMin == GL_LINEAR_MIPMAP_LINEAR and mipLevels == 1: filterMin = filterMag 
    texture = createEmptyTexture(**locals())
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, gltype, img)
    # we should do some verification if the texture is as it should be
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT)
    #int(GL_RGBA)
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_RED_TYPE)
    #int(GL_UNSIGNED_NORMALIZED)
    #glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_RED_SIZE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture

def createEmptyTexture(width, height, mipLevels=1, wrap=GL_MIRRORED_REPEAT, filterMag=GL_LINEAR, filterMin=GL_LINEAR_MIPMAP_LINEAR, gltype=GL_UNSIGNED_BYTE, isf=GL_RGBA8,**kwargs):
    ''' allocate space on the gpu for a texture, but do not fill it with anything '''
    if mipLevels <= 0:
        # Use max
        size = max(width, width)
        mipLevels = int(math.floor(math.log(size) / math.log(2.0))) + 1
    texture = glGenTextures(1)                           # setup our texture
    glBindTexture(GL_TEXTURE_2D, texture)                # bind texture
    glTexStorage2D(GL_TEXTURE_2D, mipLevels, isf, width, height)
    # set the parameters for filtering
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterMin)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMag)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture

def updateTexture(texture, image):
    ''' load new data into an existing texture '''
    img, height, width, channels, format, gltype, isf = prepareImage(image)
    glBindTexture(GL_TEXTURE_2D, texture)
    # we should do some verification if the texture is as it should be
    #width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    #height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
    #texFormat = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT)
    #texType = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_TYPE)
    #if ix == width and iy == height and format == texFormat and gltype == texType:
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, gltype, img)
    #else:
    #   pass
    glBindTexture(GL_TEXTURE_2D, 0)

# Thought about moving these two to shader module, but materials module will want them as well
def attachTexture(texture, shader, index):
    ''' attach a texture to a shader sampler2d with defalut name "tex#" '''
    attachTextureNamed(texture, shader, index, 'tex%s'%index)
# Thought about moving these two to shader module, but materials module will want them as well
def attachTextureNamed(texture, shader, index, samplerName):
    ''' attach a texture to a shader sampler2d '''
    GL.glActiveTexture(getattr(GL, 'GL_TEXTURE%s'%index))       # make texture register idx active
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)                 # bind texture to register idx
    texLoc = GL.glGetUniformLocation(shader, samplerName)       # get location of our texture
    GL.glUniform1i(texLoc, index)                               # connect location to register idx

def createWarp(width, height, gltype=GL_UNSIGNED_BYTE,wrap=GL_MIRRORED_REPEAT,filterMin=GL_LINEAR_MIPMAP_LINEAR,filterMag=GL_LINEAR, levelCount=1):
    ''' DEPRECATED - use createEmptyTexture for color and depth and create fbos yourself'''
    texture = glGenTextures(1)                           # setup our texture
    glBindTexture(GL_TEXTURE_2D, texture)                # bind texture
    levelRes = np.array([width, height], int)
    for level in range(levelCount):
        glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA, levelRes[0], levelRes[1], 0, GL_RGBA, gltype, None)   # Allocate memory
        levelRes = np.maximum(levelRes / 2, 1).astype(int)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterMin)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMag)
    glBindTexture(GL_TEXTURE_2D, 0)

    ## create a renderbuffer object to store depth info
    #depthBuffer = glGenRenderbuffers(1)
    #glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer)
    #glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    #glBindRenderbuffer(GL_RENDERBUFFER, 0)

    # use depth texture instead because we can then read it in a probably less not good way
    depthMap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    levelRes = np.array([width, height], int)
    for level in range(levelCount):
        glTexImage2D(GL_TEXTURE_2D, level, GL_DEPTH_COMPONENT, levelRes[0], levelRes[1], 0, GL_DEPTH_COMPONENT, GL_INT, None)   # Allocate memory
        levelRes = np.maximum(levelRes / 2, 1).astype(int)
    glBindTexture(GL_TEXTURE_2D, 0)

    # setup frame buffer
    fbos = []
    for level in range(levelCount):
        frameBuffer = glGenFramebuffers(1)                                                              # Create frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer)                                                  # Bind our frame buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, level)        # Attach texture to frame buffer
        #glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer)    # Attach render buffer to depth buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, level); # Attach depth texture
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        fbos.append(frameBuffer)
    
    return texture, (fbos,width,height), depthMap

def readTexture(texture, level):
    '''Get pixel values from framebuffer to numpy array'''
    glBindTexture(GL_TEXTURE_2D, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_HEIGHT)
    #format = glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_INTERNAL_FORMAT)
    #gltype = glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_RED_TYPE)
    format = GL_RGBA
    gltype = GL_UNSIGNED_BYTE
    stringArry = glGetTexImage(GL_TEXTURE_2D,level,format,gltype)     # read the texture    
    arry = np.fromstring(stringArry,glTypeToNumpy[gltype])              # convert back to numbers
    arry = np.reshape(arry,(height,width,glFormatChannels[format]))     # reshape our array to right dimensions
    arry = np.flipud(arry)                                              # openGL and openCV start images at bottom and top respectively, so flip it
    if glFormatChannels[format] > 2:                                    # swap red and blue channel
        temp = np.zeros_like(arry)
        np.copyto(temp, arry)
        temp[:,:,0] = arry[:,:,2]
        temp[:,:,2] = arry[:,:,0]
        arry=temp
    glBindTexture(GL_TEXTURE_2D, 0)
    return arry

def loadImage(imgFile):
    ''' Load an image from a file '''
    filename, extension = os.path.splitext(imgFile)
    if extension == '.npy':
        image = np.load(imgFile)
    elif extension == '.png':
        image = cv2.imread(imgFile, -1)
    else:
        raise ValueError('File gltype unknown')
    return image