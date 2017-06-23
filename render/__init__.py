#!/usr/bin/env python
''' Classes and methods for managing the render pipeline

David Dunn
Jun 2017 - created

ALL UNITS ARE IN METRIC
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '0.1'
__all__ = []

import OpenGL.GL as GL
from OpenGL.GL import *
import numpy as np
import dGraph.ui as dgui
import dGraph.textures as dgt
import cv2

def readFramebuffer(x, y, width, height, format, gltype=GL_UNSIGNED_BYTE):
    '''Get pixel values from framebuffer to numpy array'''
    stringArry = glReadPixels(x,y,width,height,format,gltype)        # read the buffer
    arry = np.fromstring(stringArry,dgt.glTypeToNumpy[gltype])              # convert back to numbers
    arry = np.reshape(arry,(height,width,dgt.glFormatChannels[format]))     # reshape our array to right dimensions
    arry = np.flipud(arry)                                              # openGL and openCV start images at bottom and top respectively, so flip it
    if dgt.glFormatChannels[format] > 2:                                    # swap red and blue channel
        temp = np.zeros_like(arry)
        np.copyto(temp, arry)
        temp[:,:,0] = arry[:,:,2]
        temp[:,:,2] = arry[:,:,0]
        arry=temp
    return arry

def createDepthRenderBuffer(width, height):
    depthBuffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)
    return depthBuffer

class RenderGraph(dict):
    '''An object representing a renderable view of one or more scene graphs
    self - the renderGraph
    _display - what display the view is rendered for
    _window - the window containing the OpenGL context for the view
    _width, _height - width and height of the view
    '''
    def __init__(self, name, file=None):
        self._name = name
        self.classifier = 'renderGraph'
        self._width = None
        self._height = None
        self.scenes = set()
        self.shaders = set()
        self.windows = set()        # a renderGraph can be displayed in multiple windows
        self.displays = set()
        self.frameBuffers = set()                               # the set of all framebuffers in the graph
        self.frameBuffer = self.add(FrameBuffer('screenFB'))    # the frameBuffer which is directly connected (rendered first)
    @property
    def name(self):     # a read only attribute
        return self._name
    def add(self, member):
        if member.name in self:
            raise ValueError('Member with name: %s already exists in render graph.'%member.name)
        else:
            self[member.name] = member
            # categorize them
            if member.classifier == 'scene':
                self.scenes.add(member.name)
            elif member.classifier == 'shader':
                self.shaders.add(member.name)
            elif member.classifier == 'frameBuffer':
                self.frameBuffers.add(member.name)
            elif member.classifier == 'display':
                self.displays.add(member.name)
            elif member.classifier == 'window':
                self._addWindow(member)
                self.windows.add(member.name)
        return member
    @property
    def width(self):
        if self._width is None:
            self.calcSize()
        return self._width
    @property
    def height(self):
        if self._height is None:
            self.calcSize()
        return self._height
    def calcSize(self):
        width, height = (0, 0)
        for win in self.windows:
            w, h = dgui.get_window_size(self[win])
            width = max(width, w)
            height = max(height, h)
        self._width = None if width == 0 else width
        self._height = None if height == 0 else height
    def _addWindow(self, window):
        id = dgui.get_window_id(window)
        other = dgui.WINDOWSTACKS.get(id, None)
        if other is not None:
            other.removeWindow(window)
        dgui.WINDOWSTACKS[id] = self
        return window
    def removeWindow(self, window):
        self._windows.remove(window)
        id = dgui.get_window_id(window)
        dgui.WINDOWSTACKS.pop(id)
    def graphicsCardInit(self):
        ''' compile shaders and create VBOs and such '''
        sceneGraphSet = set()
        sceneGraphSet.update(self.frameBuffer.setup(self.width, self.height))
        for sceneGraph in sceneGraphSet:
            for objName in sceneGraph.shapes:                # convert the renderable objects in the scene
                if sceneGraph[objName].renderable:
                    sceneGraph[objName].generateVBO()
    def render(self):
        ''' renders the our framebuffer to screen '''
        self.frameBuffer.render(None)

class FrameBuffer(object):
    ''' An object to manage openGL framebuffers '''
    def __init__(self, name, onScreen=True, depthIsTexture=False,mipLevels=1):
        self._name = name
        self.classifier = 'frameBuffer'
        self.onScreen = onScreen
        self.fbos = [0] if self.onScreen else []
        self.clear = True
        self.depthIsTexture = depthIsTexture
        self.subimages = []
        self.textures = {}  # 'Name': GLid
        self._setup = False
        self.mipLevels = mipLevels
    @property
    def name(self):     # a read only attribute
        return self._name
    def rgba(self):         # should change to __getattr__ ???
        return self.textures['rgba']
    def depth(self):        # should change to __getattr__ ???
        return self.textures['depth']
    def setResolution(self, width, height):
        self.resolution = np.array((width, height))
    def connectInput(self, subimage, posWidth=0, posHeight=0, width=1, height=1):
        location = np.array(((posWidth, posHeight), (width, height)))    # these are fractions of the image
        self.subimages.append([subimage, location])
    def createFBO(self, level=0):
        frameBuffer = glGenFramebuffers(1)                                                              # Create frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer)                                                  # Bind our frame buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rgba(), level)     # Attach texture to frame buffer
        if self.depthIsTexture:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth(), level) # Attach depth texture
        else:
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth())# Attach render buffer to depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return frameBuffer

    def setup(self, width, height):
        ''' create the FBO and textures and setup all inputs '''
        if self._setup:
            return set()
        self.setResolution(width,height)
        if not self.onScreen:
            rgba = dgt.createEmptyTexture(width,height,mipLevels=self.mipLevels)
            self.textures['rgba'] = rgba
            if self.depthIsTexture:
                depth = dgt.createEmptyTexture(width,height,mipLevels=self.mipLevels,isf=GL_DEPTH_COMPONENT)
            else:
                depth = createDepthRenderBuffer(width,height)
            self.textures['depth'] = depth
            for level in range(self.mipLevels):
                self.fbos.append(self.createFBO(level))
        sceneGraphSet = set()
        for subimage, location in self.subimages:
            sceneGraphSet.update(subimage.setup(width,height))
        self._setup = True
        return sceneGraphSet

    def render(self,resetFBO,mipLevel=0):
        ''' Renders the subimages to the frame buffers '''
        #print('%s entering render. %s'%(self.__class__, self._name))
        for level, fbo in enumerate(self.fbos):
            #print('%s setting fbo to %s'%(self._name, fbo))
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)          # Render to ourselves
            if self.clear:
                #print('%s is clearing DEPTH AND COLOR'%self.name)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for subimage, location in self.subimages:
                subimage.render(self, level)
            '''
            data = readFramebuffer(0, 0, self.resolution[0], self.resolution[1], GL_RGBA, GL_UNSIGNED_BYTE)
            cv2.imshow('%s:%s'%(self._name,fbo),data)
            cv2.waitKey()
            try:
                samplerName, texture = ('rgba',self.rgba())
                data = dgt.readTexture(texture, 0)
                cv2.imshow('%s:%s:%s'%(self._name,samplerName,texture),data)
                cv2.waitKey()
            except KeyError:
                pass
            '''
            #print('%s setting fbo to %s'%(self._name, resetFBO))
        glBindFramebuffer(GL_FRAMEBUFFER, 0 if resetFBO is None else resetFBO.fbos[min(mipLevel,len(self.fbos)-1)])
        #print('%s leaving render. %s'%(self.__class__, self._name))
        # should probably return the set of upstream nodes (and thiers) and add ourselves to avoid duplicate rendering in a single frame
        # we could then have a flag for frameRendered or frameComplete that gets checked at beginning of method and reset with new frame




