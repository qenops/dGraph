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
__all__ = ["Material", "Lambert", "Blinn", "Reflective"]

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import shaders
from math import sqrt
from numpy.linalg import norm
from numpy import dot, vdot
#import cv2

_shaderHeader = '''
#version 330
#ifdef GL_ES
    precision highp float;
#endif
'''

class Material(object):
    ''' A definition of a lighting model to define the illumination of a surface
        ???? Accepts Shadows
        Render point (given a position and normal, will return the color of specified point)
    '''
    _materialList = {}                                          # store all existing materials here
    def __new__(cls, name, *args, **kwargs):                    # keeps track of all existing materials
        if name in cls._materialList.keys():
            if not isinstance(cls._materialList[name],cls):     # do some type checking to prevent mixed results
                raise TypeError('Material of name "%s" already exists and is type: %s'%(name, type(cls._materialList[name])))
        else:
            cls._materialList[name] = super(Material, cls).__new__(cls, name, *args, **kwargs)
        return cls._materialList[name]
    def __init__(self, name, ambient=(.3,.3,.3), amb_coeff=1, **kwargs):      
        self._name = name
        self._ambient = Plug()
        self.setAmbient(ambient)
        self.setAmbientCoefficient(amb_coeff)
        self.shader = None
        self._vertexShader = """
uniform mat4 fullMatrix;
uniform mat4 modelViewMatrix;
uniform mat3 normalMatrix;
in vec3 position;
in vec3 normal;
out vec3 fragNormal;
out vec4 fragPosition;
void main()
{
    gl_Position = fullMatrix * vec4(position, 1.0);
    fragNormal = normalize(normalMatrix * normal);
    fragPosition = modelViewMatrix * vec4(position, 1.0);
}
            """
        self._fragmentShader = """
void main()
{
    gl_FragColor = vec4(%ff, %ff, %ff, %ff);
}
            """
    @property
    def name(self):
        return self._name
    @property
    def vertexShader(self):
        '''The output of the vertex shader is clip coordinates (not viewport coordinates) - OpenGL still performs the "divide by w" step automatically.'''
        return '%s%s'%(_shaderHeader, self._vertexShader)
    @property
    def fragmentShader(self):
        a = np.ones(4)
        a[:self._ambient.shape[0]] = self._ambient*self._amb_coeff
        return '%s%s'%(_shaderHeader,self._fragmentShader%tuple(a))
    def compileShader(self):
        self.shader = shaders.compileProgram(
            shaders.compileShader(self.vertexShader, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.fragmentShader, GL.GL_FRAGMENT_SHADER)
        )
    def setAmbient(self, ambient):
        self._ambient = np.array(ambient)
    def setAmbientCoefficient(self, amb_coeff):
        self._amb_coeff = amb_coeff
    def render(self, point, normal, **kwargs):
        return self._ambient*self._amb_coeff

class Test(Material):
    ''' A test material class that is lighting agnostic and displays all faces distinctly '''
    def __init__(self,name, **kwargs):
        super(Test, self).__init__(name, **kwargs)
        self._fragmentShader = '''
smooth in vec3 fragNormal;      // normal in camera space
smooth in vec4 fragPosition;    // position in camera space

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;

void main()
{
  //FragColor = fragPosition;
  //FragColor = vec4(fragNormal, 1.0);
  //FragColor = vec4(fragPosition.z/-10,fragPosition.z/-10,fragPosition.z/-10,1f);
  //FragColor = vec4(gl_FragCoord.z*1000,gl_FragCoord.z*1000,gl_FragCoord.z*1000,1f);
  int ID = gl_PrimitiveID + 1;
  FragColor = vec4(mod(ID,15)/15,mod(ID,7)/7,mod(ID,3)/3, 1f);
}'''
    @property
    def fragmentShader(self):
        return '%s%s'%(_shaderHeader,self._fragmentShader)

class Lambert(Material):
    ''' A material class based on the Lambert illumination model
        Diffuse Color: (1, 1, 1)
        Diffuse Coeff: 0-1
    '''
    def __init__(self, name, diffuse=(0,0,0), diff_coeff=0, **kwargs):        
        super(Lambert, self).__init__(name, **kwargs)
        #self._shadows = shadows
        self.setDiffuse(diffuse)
        self.setDiffuseCoefficient(diff_coeff)
        self._fragmentShader = '''
smooth in vec3 fragNormal;      // normal in camera space
smooth in vec4 fragPosition;    // position in camera space

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;

// Lights
struct Light {
    vec3 color;
    vec3 position;
};  

float lambert(vec3 N, vec3 L)
{
  vec3 nrmN = normalize(N);
  vec3 nrmL = normalize(L);
  float result = dot(nrmN, nrmL);
  return max(result, 0.0);
}

void main()
{
  Light light;
  light.position = vec3(2f,3f,4f);
  light.color = vec3(%sf, %sf, %sf);

  vec3 L = normalize(light.position.xyz - fragPosition.xyz);   
  vec3 Idiff = light.color * lambert(fragNormal,L);  
  Idiff = clamp(Idiff, 0.0, 1.0); 
  Idiff = Idiff + vec3(%sf, %sf, %sf);
  FragColor = vec4(Idiff, 1.0);
  //FragColor = vec4(fragNormal, 1.0);
}'''

    @property
    def fragmentShader(self):
        diffused = self._fragmentShader%(tuple(self._diffuse*self._diff_coeff)+('%s','%s','%s'))
        diffused = diffused%tuple(self._ambient*self._amb_coeff)
        return '%s%s'%(_shaderHeader,diffused)
        #return self._fragmentShader
    def setDiffuse(self, diffuse):
        self._diffuse = np.array(diffuse)
    def setDiffuseCoefficient(self, diff_coeff):
        self._diff_coeff = diff_coeff
    def render(self, point, normal, uv=None, **kwargs):
        global shadows, epsilon
        ''' BRDF render algorithm goes here '''  
        lightList = Light._lightList
        myDiffuse = self._diffuse*self._diff_coeff
        colorList = []
        if (myDiffuse==0).all():
            colorList.append(myDiffuse)
        else:
            for light in lightList.values():
                lightPos = xm.getTranslate(light.worldMatrix)
                vector = Ray._calcVector(point,lightPos)
                dist = sqrt(vdot(vector,vector))
                if shadows:
                    ray = Ray(point+epsilon*normal, vector)
                    intersections = ray.evaluate(light.getScene())
                    between = [i for i in intersections if i['distance'] < dist]
                    if between:
                        colorList.append(np.array([0,0,0]))
                        continue
                vector = vector/norm(vector)
                colorList.append(myDiffuse*light.illumination(dist)*max(0,dot(normal, vector)))
        return sum(colorList)/float(len(colorList))+super(Lambert, self).render(point,normal)   # add sum of lights and ambient together
        
class Blinn(Lambert):
    ''' A material class based on the Phong illumination model
        Specular Color: (1, 1, 1)
        Specular Coeff: 0-1
        Specular Power: 0-1
    '''
    def __init__(self, name, specular=(0,0,0), spec_coeff=0, spec_power=0, **kwargs):        
        super(Blinn, self).__init__(name, **kwargs)
        self.setSpecular(specular)
        self.setSpecularCoefficient(spec_coeff)
        self.setSpecularPower(spec_power)
    def setSpecular(self, specular):
        self._specular = np.array(specular)
    def setSpecularCoefficient(self, spec_coeff):
        self._spec_coeff = spec_coeff
    def setSpecularPower(self, spec_power):
        self._spec_power = spec_power
    def render(self, point, normal, viewVector, uv=None, **kwargs):
        global shadows, epsilon
        ''' BRDF render algorithm goes here '''
        lightList = Light._lightList
        mySpec = self._specular*self._spec_coeff
        colorList = []
        if (mySpec==0).all():
            colorList.append(mySpec)
        else:
            for light in lightList.values():
                lightPos = xm.getTranslate(light.worldMatrix)
                vector = Ray._calcVector(point,lightPos)
                dist = sqrt(vdot(vector,vector))
                if shadows:
                    ray = Ray(point+epsilon*normal, vector)
                    intersections = ray.evaluate(light.getScene())
                    between = [i for i in intersections if i['distance'] < dist]
                    if between:
                        colorList.append(np.array([0,0,0]))
                        continue
                vector = vector/norm(vector)        
                h = vector-viewVector/norm(vector-viewVector)   # Blinn-Phong equation
                h = h/norm(h)
                blinn = max(0,dot(h,normal))**self._spec_power
                colorList.append(mySpec*light.illumination(dist)*blinn)
        return sum(colorList)/float(len(colorList))+super(Blinn, self).render(point,normal)   # add sum of lights and lambert together

class Reflective(Blinn):
    ''' A reflective material class based on the Phong illumination model
        Reflectance Strength: 0-1
    '''
    def __init__(self, name, reflectance=0, **kwargs):        
        super(Reflective, self).__init__(name, **kwargs)
        self.setReflectance(reflectance)
    def setReflectance(self, reflectance):
        self._reflectance = reflectance
    def render(self, point, normal, viewVector, world, uv=None, **kwargs):
        #L = (1-r)(La + Ld + Ls) + rLm
        ''' Recursive raytracing algorithm goes here '''
        # verify current reflectDepth, if more than 1, stop reflecting, just return blinn portion
        global reflectDepth
        if reflectDepth > reflectionsMax:
            #print "max reflection depth achieved"
            return super(Reflective, self).render(point,normal,viewVector)
        # calculate new ray (reflect viewVector across normal)  r=v?2(v?n)n
        refVect = viewVector-2*(dot(viewVector,normal))*normal
        ray = Ray(point+epsilon*normal, refVect)
        # render the new ray
        reflectDepth += 1
        color = ray.render(world)
        reflectDepth -= 1
        return (self._reflectance*color)+((1.0-self._reflectance)*super(Reflective, self).render(point,normal,viewVector))   # add weighted reflected and weighted blinn together
