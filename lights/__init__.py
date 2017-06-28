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
import dGraph.shaders as dgshdr
import numpy as np
from numpy.linalg import norm

class Light(object):
    ''' A world object that casts light 
        Intensity
        Color
    '''    
    def __init__(self, intensity=(1,1,1), **kwargs):      
        super(Light, self).__init__(**kwargs)
        self._intensity = np.array(intensity, np.float32)

    def fragmentShader(self, index):
        pass

    def pushToShader(self, index, shader):
        pass
 

class PointLight(Light):
    ''' A light with falloff '''
    def __init__(self, position = (0,0,0), **kwargs):      
        super(PointLight, self).__init__(**kwargs)
        self._position = np.array(position, np.float32)

    def fragmentShader(self, index):
        return '''
uniform vec3 light{index}_intensity;
uniform vec3 light{index}_position;

vec3 getLightDirection{index}(vec3 worldLocation) {{
    return normalize(light{index}_position - worldLocation);
}}

vec3 getLightIntensity{index}(vec3 worldLocation) {{
    return light{index}_intensity;
}}

'''.format(index = index)

    def pushToShader(self, index, shader):
        #import pdb; pdb.set_trace();
        dgshdr.setUniform(shader, 'light{index}_intensity'.format(index=index), np.array(self._intensity, np.float32))
        dgshdr.setUniform(shader, 'light{index}_position'.format(index=index), np.array(self._position, np.float32))

        

class DirectionLight(Light):
    ''' A light where position doesn't matter, only a direction vector '''
    def __init__(self, direction=(0.,0.,1.0), **kwargs):      
        super(DirectionLight, self).__init__(**kwargs)
        self._direction = np.array(direction, np.float32)

    def fragmentShader(self, index):
        return '''
uniform vec3 light{index}_intensity;
uniform vec3 light{index}_direction;

vec3 getLightDirection{index}(vec3 worldLocation) {{
    return normalize(light{index}_direction);
}}

vec3 getLightIntensity{index}(vec3 worldLocation) {{
    return light{index}_intensity;
}}


'''.format(index = index)

    def pushToShader(self, index, shader):
        #import pdb; pdb.set_trace();
        dgshdr.setUniform(shader, 'light{index}_intensity'.format(index=index), np.array(self._intensity, np.float32))
        dgshdr.setUniform(shader, 'light{index}_direction'.format(index=index), np.array(self._direction, np.float32))
