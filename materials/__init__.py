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
        self.specularTexture = None
        self.bumpTexture = None


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
        if self.specularTexture:
            code += "uniform sampler2D {name}_specular;\n".format(name=name)
        if self.bumpTexture:
            code += "uniform sampler2D {name}_bump;\n".format(name=name)
        code += "\n\n";
        return code

    def fragmentShaderShading(self, name, scene):
        code = """
vec3 {name}_shading(
	const vec3 position, 
	const vec3 outDirectionCS,
	vec3 normal, vec3 tangent, vec3 bitangent,
	const vec2 texCoord)
{{"""
        if self.diffuseTexture:
            code += '\n vec3 diffuseTexSample = texture({name}_diffuse, texCoord).rgb;\n'
        else:
            code += '\n vec3 diffuseTexSample = vec3(1);\n'
        if self.specularTexture:
            code += '\n vec3 specularTexSample = texture({name}_specular, texCoord).rgb;\n'
        else:
            code += '\n vec3 specularTexSample = vec3(1);\n'
        code += """
	vec3 diffuse = {name}.diffuseColor * diffuseTexSample;
	vec3 specularity = {name}.specularColor * specularTexSample;
	float glossiness = {name}.glossiness;

    mat3 tbnMatrix = transpose(mat3(tangent, bitangent, normal));
    vec3 normalTS = vec3(0, 0, 1); // Normal in Tangent-space
"""
        if self.bumpTexture:
            code += 'vec3 bumpTexSample = texture({name}_bump, texCoord).rgb;\n'
            code += 'normalTS = normalize(bumpTexSample * 2.0 - 1.0);\n' # this is just temporary (and wrong)

        code += """

    if (length(tangent) < 1e-3) {{
        // no tangents defined! Legacy mode
        normalTS = normal;
        tbnMatrix = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1); 
    }}

    vec3 outDirection = tbnMatrix * outDirectionCS;

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

    inDirection = tbnMatrix * inDirection;

    diffShade = inLight * diffuse * (max(0, dot(inDirection, normalTS)));
    specShade = inLight * specularity * pow(max(0, dot(outDirection, -reflect(inDirection, normalTS))), glossiness);
    
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
        if self.specularTexture:
            dgt.attachTextureNamed(self.specularTexture, shader, textureIndex, '{name}_specular'.format(name = name))
            textureIndex += 1
        if self.bumpTexture:
            dgt.attachTextureNamed(self.bumpTexture, shader, textureIndex, '{name}_bump'.format(name = name))
            textureIndex += 1

        return textureIndex
