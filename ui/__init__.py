#!/usr/bin/env python
'''User interface submodule for dGraph scene description module

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = []

import OpenGL.GL as GL

class RenderStack(list):
    '''An object representing a renderable view of the scene graph
    self - the renderStack
    _objects - list of objects in the view
    _cameras - list of cameras in the view
    _display - what display the view is rendered for
    _window - the window containing the OpenGL context for the view
    _width, _height - width and height of the view
    '''
    def __init__(self):
        super(RenderStack,self).__init__()
        self._cameras = []
        self._objects = {}
        self._display = None
        self._window = None
        self._width, self._height = (1920, 1080)

def graphicsCardInit(renderStack, width, height):
    ''' compile shaders and create VBOs and such '''
    sceneGraphSet = set()
    for node in renderStack:
        sceneGraphSet.update(node.setup(width, height))
    for sceneGraph in sceneGraphSet:
        for obj in sceneGraph:                                                      # convert the renderable objects in the scene
            if obj.renderable:
                print obj.name
                obj.generateVBO()

def resize_window_callback(window, w, h):
    '''Need to figure out how to track this
    what is rederStack -> window relationship??? '''
    global renderStack, cameras, width, height
    width = w if w > 1 else 2
    height = h if h > 1 else 2
    for cam in cameras:
        cam.setResolution((width/2, height))
    for node in renderStack:
        node.setup(width, height)