#!/usr/bin/env python
'''A library of materials classes to create shaders for rendering surfaces

David Dunn
Jan 2016 - Created
Jan 2017 - Redefined modules - split off warp shaders

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Material", "Lambert", "Blinn", "Reflective", "warp"]

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import shaders
from math import sqrt
from numpy.linalg import norm
from numpy import dot, vdot
import dGraph as dg
import dGraph.config as config
import dGraph.textures as dgt
import dGraph.shaders as dgshdr
#import cv2




class Material(object):
    ''' A material class based on the Phong illumination model
    '''
    def __init__(self, name = ''):      
        self.name = name
        self.classifier = 'material'

        self.diffuseColor = np.array([1.0, 1.0, 1.0])
        self.specularColor = np.array([0.0, 0.0, 0.0])
        self.glossiness = 1.0

        self.diffuseTexture = None


    @staticmethod
    def getShaderStruct():
        return """
struct Material {
    vec3 diffuseColor;
    vec3 specularColor;
    float glossiness;
};


        """

    def fragmentShaderDefinitions(self, name):
        code = "";
        code += "uniform Material {name};\n".format(name=name)
        if self.diffuseTexture:
            code += "uniform sampler2D {name}_diffuse;\n".format(name=name)
        code += "\n\n";
        return code

    def fragmentShaderShading(self, name, scene):
        code = """
vec3 {name}_shading(
	const vec3 position, 
	const vec3 outDirection,
	const vec3 normal,
	const vec2 texCoord)
{{"""
        if self.diffuseTexture:
            code += '\n vec3 diffuseTexSample = texture({name}_diffuse, texCoord).rgb;\n'
        else:
            code += '\n vec3 diffuseTexSample = vec3(1);\n'
        code += """
	vec3 diffuse = {name}.diffuseColor * diffuseTexSample;
	vec3 specularity = {name}.specularColor;
	float glossiness = 10 * {name}.glossiness;

    vec3 result = vec3(0);
    result += ambientLight * diffuse;

    vec3 diffShade;
    vec3 specShade;
    vec3 inDirection;
    vec3 inLight;
"""
        # light loop
        for i,light in enumerate(scene.lights):
            code += """
    inDirection = getLightDirection{index}(position);
    inLight = getLightIntensity{index}(position);

    diffShade = inLight * diffuse * (max(0, dot(inDirection, normal)));
    specShade = inLight * specularity * pow(max(0, dot(outDirection, -reflect(inDirection, normal))), glossiness);
    
	result += diffShade + specShade;
            """.format(index = i)


        code += """
	return result;	
}}


"""
        code = code.format(name=name)
        return code

    def pushToShader(self, name, shader, textureIndex):
        #import pdb; pdb.set_trace();
        dgshdr.setUniform(shader, '{name}.diffuseColor'.format(name=name), np.array(self.diffuseColor, np.float32))
        dgshdr.setUniform(shader, '{name}.specularColor'.format(name=name), np.array(self.specularColor, np.float32))
        dgshdr.setUniform(shader, '{name}.glossiness'.format(name=name), float(self.glossiness))
        if self.diffuseTexture:
            dgt.attachTextureNamed(self.diffuseTexture, shader, textureIndex, '{name}_diffuse'.format(name = name))
            textureIndex += 1
        return textureIndex
