#!/usr/bin/env python
'''User interface submodule for dGraph scene description module based on glfw

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = []

WINDOWSTACKS = {}       # Each window can have 1 associated renderStack
WINDOWS = []
import OpenGL.GL as GL
import dglfw as fw
from dglfw import *

class RenderStack(list):
    '''An object representing a renderable view of the scene graph
    self - the renderStack
    _objects - list of objects in the view
    _cameras - list of cameras in the view
    _display - what display the view is rendered for
    _window - the window containing the OpenGL context for the view
    _width, _height - width and height of the view
    '''
    def __init__(self, *args, **kwargs):
        super(RenderStack,self).__init__(*args, **kwargs)
        self._windows = []      # a renderStack can be displayed in multiple windows   
        self._width = None
        self._height = None
        self.cameras = []       # just for convinience if wanted
        self.objects = {}       # just for convinience if wanted
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
                    print obj.name
                    obj.generateVBO()

class Display(object):
    ''' A class that defines the physical properties of a display '''
    def __init__(self, resolution=(1080,1920), size=(.071,.126), bezel=((.005245,.005245),(.01,.01)),location=np.array(0.,0.,0.)):  # default to Samsung Note 3
        self.resolution = resolution
        self.size = size
        self.bezel = bezel
        self.location = location  # the top left corner of the display (not the bezel)
    @property
    def width(self):
        return self.resolution[1]
    @property
    def height(self):
        return self.resolution[0]
    def pixelSize(self):
        return (self.size[0]/self.resolution[0], self.size[1]/self.resolution[1])

def resize_window_callback(window, w, h):
    '''Need to figure out how to track this
    what is rederStack -> window relationship??? '''
    renderStack = WINDOWSTACKS[window]
    width = w if w > 1 else 2
    height = h if h > 1 else 2
    renderStack._width = None
    renderStack._height = None
    for cam in cameras:
        cam.setResolution((width/2, height))  # for binocular ???
    for node in renderStack:
        node.setup(renderStack.width, renderStack.height)

def get_window_id(window):
    try:
        id = WINDOWS.index(window)
    except ValueError:
        id = len(WINDOWS)
        WINDOWS.append(window)
    return id

def close_window(window):
    id = get_window_id(window)
    rs = WINDOWSTACKS.get(id, None)
    if rs is not None:
        rs.removeWindow(window)
    WINDOWS[id] = None
    fw.set_window_should_close(window, True)