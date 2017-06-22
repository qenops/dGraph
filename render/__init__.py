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
import numpy as np
import dGraph.ui as dgui
import dGraph.textures as dgt

class RenderGraph(object):
    '''An object representing a renderable view of the scene graph
    self - the renderGraph
    _display - what display the view is rendered for
    _window - the window containing the OpenGL context for the view
    _width, _height - width and height of the view
    '''
    def __init__(self, *args, **kwargs):
        super(RenderGraph, self).__init__(*args, **kwargs)
        self._windows = []      # a renderGraph can be displayed in multiple windows
        self._width = None
        self._height = None
        self.frameBuffer = FrameBuffer()
        self.scenes = []        # just for convinience if wanted
        self.shaders = {}       # just for convinience if wanted
        self.displays = []      # just for convinience if wanted
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
        for win in self._windows:
            w, h = dgui.get_window_size(win)
            width = max(width, w)
            height = max(height, h)
        self._width = None if width == 0 else width
        self._height = None if height == 0 else height
    @property
    def windows(self):
        return list(self._windows)
    def addWindow(self, window):
        self._windows.append(window)
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
            for obj in sceneGraph:                                                      # convert the renderable objects in the scene
                if obj.renderable:
                    obj.generateVBO()
    def render(self):
        ''' renders the our framebuffer to screen '''
        self.frameBuffer.render()

class FrameBuffer(object):
    ''' An object to manage openGL framebuffers '''
    def __init__(self, width=None, height=None):
        self.fbo = 0
        self.clear = True
        self.depthIsTexture = False
        self.subimages = []
        self.setResolution(width, height)
        self.textures = {}  # Name: GLid
    @property
    def rgba(self):         # should change to __getattr__
        return self.textures['rgba']
    def setResolution(self, width, height):
        self.resolution = np.array((width, height))
    def connectInput(self, subimage, posWidth=0, posHeight=0, width=1, height=1):
        location = np.array(((posWidth, posHeight), (width, height)))    # these are fractions of the image
        self.subimages.append([subimage, location])
    def createFBO(self, level=0):
        frameBuffer = glGenFramebuffers(1)                                                              # Create frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer)                                                  # Bind our frame buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.textures['rgba'], level)     # Attach texture to frame buffer
        if self.depthIsTexture:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.textures['depth'], level) # Attach depth texture
        else:
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.textures['depth'])# Attach render buffer to depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return frameBuffer
    def setup(self, width, height):
        ''' create the FBO and textures and setup all inputs '''
        self.setResolution(width,height)
        if self.fbo != 0:
            rgba = dgt.createEmptyTexture(width,height)
            self.textures['rgba'] = rgba
            if self.depthIsTexture:
                depth = dgt.createEmptyTexture(width,height,isf=GL_DEPTH_COMPONENT)
            else:
                depth = dgt.createDepthRenderBuffer(width,height)
            self.textures['depth'] = depth
        self.fbo = self.createFBO()
        sceneGraphSet = set()
        for subimage, location in self.subimages:
            sceneGraphSet.update(subimage.setup(width,height))
        return sceneGraphSet
    def render(self):
        ''' Renders the subimages to this frame buffer '''
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glid)          # Render to ourselves
        if self.clear:
            #print('Clearing %s'%self.fbo)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for subimage, location in self.subimages:
            GL.glViewport(*(self.resolution*location).flatten())                      # set the viewport to the portion we are drawing
            subimage.render()

class MipFrameBuffer(FrameBuffer):
    ''' An object to manage mipmapped openGL framebuffers '''
    def __init__(self, width=None, height=None, mipLevels=1):
        super().__init__(width,height)
        self.depthIsTexture = True
        self.mipLevels = mipLevels
        self.fbos = []
    def setup(self, width, height):
        self.setResolution(width,height)
        if self.fbo != 0:
            rgba = dgt.createEmptyTexture(width,height,mipLevels=self.mipLevels)
            self.textures['rgba'] = rgba
            if self.depthIsTexture:
                depth = dgt.createEmptyTexture(width,height,mipLevels=self.mipLevels,isf=GL_DEPTH_COMPONENT)
            else:
                depth = dgt.createDepthRenderBuffer(width,height)
            self.textures['depth'] = depth
        for level in range(self.mipLevels):
            fbos.append(self.createFBO(level))

