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