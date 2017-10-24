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
import numpy as np
#from . import dglfw as fw
#from .dglfw import *
WINDOWSTACKS = {}       # Each window can have 1 associated renderGraph
WINDOWS = []

# should look at glfw Monitor objects
class Display(object):
    ''' A class that defines the physical properties of a display AKA a monitor'''
    def __init__(self, name, monitor, bezel=None,location=(0.,0.,0.)):
        self.name = name
        self.classifier = 'display'
        self.resolution, self.colorDepth, self.fps = [np.array(a) for a in fw.get_video_mode(monitor)]
        self.glResolution = np.flipud(self.resolution)
        #print(self.resolution,self.colorDepth, self.fps)
        self.fps = 60 if self.fps == 59 else 30 if self.fps == 29 else self.fps  # fix rounding down errors
        self.size = np.array(fw.get_monitor_physical_size(monitor))/1000.
        self.screenPosition = np.array(fw.get_monitor_pos(monitor))
        self.bezel = None if bezel is None else np.array(bezel)
        self.location = None if location is None else np.array(location)  # the top left corner of the display (not the bezel)
    @property
    def width(self):
        return self.resolution[0]
    @property
    def height(self):
        return self.resolution[1]
    def pixelSize(self):
        return self.size/self.resolution

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