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

class RenderGraph(object):
    '''An object representing a renderable view of the scene graph
    self - the renderGraph
    _objects - list of objects in the view
    _cameras - list of cameras in the view
    _display - what display the view is rendered for
    _window - the window containing the OpenGL context for the view
    _width, _height - width and height of the view
    '''
    def __init__(self, *args, **kwargs):
        super(RenderGraph,self).__init__(*args, **kwargs)
        self._windows = []      # a renderGraph can be displayed in multiple windows
        self._width = None
        self._height = None
        self.cameras = []       # just for convinience if wanted
        self.objects = {}       # just for convinience if wanted
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
        width, height = (0,0)
        for win in self._windows:
            w,h = fw.get_window_size(win)
            width = max(width, w)
            height = max(height, h)
        self._width = None if width == 0 else width
        self._height = None if height == 0 else height
    @property
    def windows(self):
        return list(self._windows)
    def addWindow(self, window):
        self._windows.append(window)
        id = get_window_id(window)
        other = WINDOWSTACKS.get(id,None)
        if other is not None:
            other.removeWindow(window)
        WINDOWSTACKS[id] = self
        return window
    def removeWindow(self, window):
        self._windows.remove(window)
        id = get_window_id(window)
        WINDOWSTACKS.pop(id)

    def graphicsCardInit(self):
        ''' compile shaders and create VBOs and such '''
        sceneGraphSet = set()
        for node in self:
            sceneGraphSet.update(node.setup(self.width, self.height))
        for sceneGraph in sceneGraphSet:
            for obj in sceneGraph:                                                      # convert the renderable objects in the scene
                if obj.renderable:
                    obj.generateVBO()

class RenderObject(object):
    ''' Any object that belongs in the renderGraph
        Shaders
        Cameras
    '''
    def __init__(self, name, parent):
        self._children = []
        self._parent = None
        self.parent = parent
        self._name = name
        self._shapes = []
        self.renderable = False

class FrameBuffer(object):
    ''' An object to manage openGL framebuffers '''
    def __init__(self, glid=0):
        self.glid = glid
        self.clear = True
        self.inputs = []
    def render(self,width,height):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glid)          # Render to our parentFrameBuffer, not screen
        if self.clear:
            #print('Clearing %s'%self.glid)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for subimage, location in self.inputs:
            GL.glViewport()                      # set the viewport to the portion we are drawing
            subimage.render(location[3],location[4])
        
    def connectInput(self, other):
        posWidth, posHeight, width, height
    
