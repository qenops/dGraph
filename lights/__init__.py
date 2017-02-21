#!/usr/bin/env python
'''Light submodule for dGraph scene description module

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Light", "PointLight", "DirectionLight"]

from dGraph import *
import numpy as np
from numpy.linalg import norm

class Light(WorldObject):
    ''' A world object that casts light 
        Intensity
        Color
    '''
    _lightList = {}                                         # store all existing lights here
    def __new__(cls, name, *args, **kwargs):                # use for creating new materials - keeps track of all existing materials
        if name in cls._lightList.keys():
            if not isinstance(cls._lightList[name],cls):    # do some type checking to prevent mixed results 
                raise TypeError('Light of name "%s" already exists and is type: %s'%(name, type(cls._lightList[name])))
        else:
            cls._lightList[name] = super(Light, cls).__new__(cls, name, *args, **kwargs)
        return cls._lightList[name]
    @classmethod
    def allLights(cls):
        return cls._lightList
    def __init__(self, name, parent, color=(1,1,1), intensity=1, **kwargs):      
        super(Light, self).__init__(name, parent)
        self._color = np.array(color)
        self._intensity = intensity
    @property
    def name(self):
        return self.name
    def illumination(self, distance):
        return self._color*self._intensity
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 position;
};
'''

class PointLight(Light):
    ''' A light with falloff '''
    def __init__(self, name, parent, falloff=0, **kwargs):      
        super(PointLight, self).__init__(name, parent)
        self._falloff = falloff
    def illumination(self, distance):
        ''' need to add calculation for falloff - inverse square law or something equiv '''
        return self._color*self._intensity
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 position;
    float falloff;
};
'''

class DirectionLight(Light):
    ''' A light where position doesn't matter, only a direction vector '''
    def __init__(self, name, parent, direction=(1.,0.,0.), **kwargs):      
        super(PointLight, self).__init__(name, parent)
        self._direction = np.array(direction)
    @property
    def direction(self):
        return self._direction/norm(self._direction)
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 direction;
};
'''