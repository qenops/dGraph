#!/usr/bin/env python
# pylint: disable=bad-whitespace, line-too-long
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

from dGraph.ui import dglfw as fw
from dGraph.ui.dglfw import *
#from . import dglfw as fw
#from .dglfw import *
WINDOWSTACKS = {}       # Each window can have 1 associated renderGraph
WINDOWS = []

class Display(object):
    ''' A class that defines the physical properties of a display '''
    def __init__(self, resolution=(1080,1920), size=(.071,.126), bezel=((.005245,.005245),(.01,.01)),location=(0.,0.,0.)):  # default to Samsung Note 3
        self.resolution = resolution
        self.size = size
        self.bezel = bezel
        self.location = location  # the top left corner of the display (not the bezel)
    @property
    def width(self):
        return self.resolution[0]
    @property
    def height(self):
        return self.resolution[1]
    def pixelSize(self):
        return (self.size[0]/self.resolution[0], self.size[1]/self.resolution[1])

def resize_window_callback(window, w, h):
    ''' BROKEN - DON'T USE
    Need to figure out how to track this
    what is rederStack -> window relationship??? '''
    renderGraph = WINDOWSTACKS[window]
    width = w if w > 1 else 2
    height = h if h > 1 else 2
    renderGraph._width = None
    renderGraph._height = None
    for cam in cameras:
        cam.setResolution((width/2, height))  # for binocular ???
    for node in renderGraph:
        node.setup(renderGraph.width, renderGraph.height)

def get_window_id(window):
    try:
        id = WINDOWS.index(window)
    except ValueError:
        id = len(WINDOWS)
        WINDOWS.append(window)
    return id

def close_window(window):
    id = get_window_id(window)
    rg = WINDOWSTACKS.get(id, None)
    if rg is not None:
        rg.removeWindow(window)
    WINDOWS[id] = None
    fw.set_window_should_close(window, True)