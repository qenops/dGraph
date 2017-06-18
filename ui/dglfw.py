#!/usr/bin/env python
'''GLFW interface submodule for dGraph scene description module

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'

_key_callbacks = {}

import glfw
from glfw import *

def init():
    glfw.set_error_callback(error_callback)
    if not glfw.init():
        exit(1)

def error_callback(error, description):
    print("Error: %s\n" % description)

def add_key_callback(function, key, action=glfw.PRESS, mods=0x0000, *args, **kwargs):
    ''' key callbacks are stored as dict[(key, action, mods)] = (function, args, kwargs)
    note that function must always accept the window as first argument'''
    _key_callbacks[(key,action,mods)] = (function, args, kwargs)

def key_callback(window, key, scancode, action,  mods):
    if (key,action,mods) in _key_callbacks.keys():
        function, args, kwargs = _key_callbacks[(key,action,mods)]
        function(window, *args, **kwargs)

class Joystick(int):
    def __init__(self, *args, **kwargs):
        super(Joystick,self).__init__(*args, **kwargs)
        self.leftTrigger = False
        self.rightTrigger = False
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.L = False
        self.R = False

def find_joysticks():
    joylist = []
    for joy in range(15):
        if glfw.joystick_present(joy):
            axes = glfw.get_joystick_axes(joy)
            sum = 2.
            for i in range(axes[1]):
                sum += axes[0][i]
            if abs(sum) > .000001:
                print('Found Joy %d'%(joy+1))
                joylist.append(Joystick(joy))
    return joylist

def get_joy_axes(joy):
    axes = glfw.get_joystick_axes(joy)
    toReturn = [0.]*axes[1]
    for i in range(axes[1]):
        toReturn[i] = axes[0][i]
    return toReturn

def get_joy_buttons(joy):
    buttons = glfw.get_joystick_buttons(joy)
    toReturn = [0.]*buttons[1]
    for i in range(buttons[1]):
        toReturn[i] = buttons[0][i]
    return toReturn

def open_window(title, posX, posY, width, height, share=None, key_callback=key_callback):
    glfw.window_hint(glfw.VISIBLE, False)
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.SAMPLES, 8)
    window = glfw.create_window(width, height, title, None, share)
    if not window:
        return None
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # doesn't seem to be helping fix vsync - actually turning it to 0 does seem to help
    glfw.set_window_pos(window, posX, posY)
    glfw.show_window(window)
    glfw.set_key_callback(window, key_callback)
    return window


'''
# figure out location of screen
import glfw
glfw.init()
glfw.window_hint(glfw.DECORATED, False)
location = (5176, 1476)
size = (1050, 1260)
width = size[0]
height = size[1]
posX = location[0]
posY = location[1]
title = 'Test'
share = None
window = glfw.create_window(width, height, title, None, share)
glfw.make_context_current(window)
glfw.swap_interval(1)
glfw.set_window_pos(window, posX, posY)
glfw.show_window(window)
def key_callback(window, key, scancode, action,  mods):
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
glfw.set_key_callback(window, key_callback)
glfw.swap_buffers(window)
glfw.swap_buffers(window)
glfw.swap_buffers(window)
location = glfw.get_window_pos(window)
size = glfw.get_window_size(window)
#size = glfw.set_window_size(window2, width, height)

location = (6226, 1476)
size = (998, 1260)
width = size[0]
height = size[1]
posX = location[0]
posY = location[1]
title = 'Test 2'
share = None
window2 = glfw.create_window(width, height, title, None, share)
glfw.make_context_current(window2)
glfw.swap_interval(1)
glfw.set_window_pos(window2, posX, posY)
glfw.show_window(window2)
def key_callback(window2, key, scancode, action,  mods):
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window2, True)
glfw.set_key_callback(window2, key_callback)
glfw.swap_buffers(window2)
glfw.swap_buffers(window2)
glfw.swap_buffers(window2)
size = glfw.get_window_size(window2)
location = glfw.get_window_pos(window2)
'''