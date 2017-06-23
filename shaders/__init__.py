#!/usr/bin/env python
# pylint: disable=bad-whitespace, line-too-long
'''A library of warp shaders for manipulating images

David Dunn
Jan 2017 - Created by splitting off from main materials file

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Warp", "Contrast", "Blur", "Convolution", "Lookup"]

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import *
from OpenGL.GL import shaders
import ctypes, math
import dGraph.textures as dgt
import dGraph.config as config
import cv2

def setUniform(shader, name, value):
    ''' Sets a uniform (not sampler, just like float and stuff) '''
    location = glGetUniformLocation(shader, name)
    if location < 0:
        return
    if type(value) == int:
        glProgramUniform1i(shader, location, value)

class Warp(object):
    ''' A warp class that takes an image and alters it in some manner '''
    _warpList = {}                                          # store all existing materials here
    def __new__(cls, name, *args, **kwargs):                # keeps track of all existing materials
        if name in cls._warpList.keys():
            if not isinstance(cls._warpList[name],cls):     # do some type checking to prevent mixed results
                raise TypeError('Material of name "%s" already exists and is type: %s'%(name, type(cls._materialList[name])))
        else:
            cls._warpList[name] = super(Warp, cls).__new__(cls)
            cls._warpList[name].__init__(name, *args, **kwargs) # this is ugly. __new__ should not be used for this at all
        return cls._warpList[name]
    def __init__(self, name, **kwargs):      
        self._name = name
        self.classifier = 'shader'
        self._setup = False
        self._internalTextureData = {}      # named dict containing internal texture data
        self._textures = {}                 # named textures for attachment 
        self._upstreamBuffers = set()
        self.shader = None
        self._vertexShader = '''
in vec3 position;
in vec2 texCoord;
out vec2 fragTexCoord;
void main() {
    gl_Position = vec4(position,1.0f);
    fragTexCoord = texCoord;
}
'''
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D texRGBA;
// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
    FragColor = texture2D( texRGBA, fragTexCoord );
    //FragColor = vec4(fragTexCoord.x,fragTexCoord.y,1f,1f); 
    //vec4 col = texture2D( texRGBA, fragTexCoord );
    //FragColor = vec4(col.a, col.a, col.a, col.a);
}
'''
    @property
    def name(self):     # a read only attribute
        return self._name
    @property
    def vertexShader(self):
        '''The output of the vertex shader is clip coordinates (not viewport coordinates) - OpenGL still performs the "divide by w" step automatically.'''
        return '%s%s'%(config.shaderHeader, self._vertexShader)
    @property
    def fragmentShader(self):
        return '%s%s'%(config.shaderHeader,self._fragmentShader)
    def compileShader(self):
        self.shader = shaders.compileProgram(
            shaders.compileShader(self.vertexShader, GL_VERTEX_SHADER),
            shaders.compileShader(self.fragmentShader, GL_FRAGMENT_SHADER),
            )
    def connectInput(self, textureFunction, samplerName=None, overwrite=False):
        if samplerName is None:
            samplerName = '%d'%len(self._textures)
        if samplerName in self._textures and not overwrite:
            raise ValueError('Texture with sampler name: %s already attached to shader %s.'%(samplerNname,self.name))
        self._upstreamBuffers.add(textureFunction.__self__)
        self._textures[samplerName] = textureFunction       # store the function, so we can call it later to get the texture after it is created
    def setup(self, width, height):
        ''' Setup our geometry and compile our shaders '''
        if self._setup:
            return set()
        self._width = width
        self._height = height
        #self._warpList = []                  # clear textures for resizing
        if not hasattr(self, 'vertexArray'):
            self.setupGeo()
        sceneGraphSet = set()
        for frameBuffer in self._upstreamBuffers:
            sceneGraphSet.update(frameBuffer.setup(width, height))
        for samplerName, textureFunction in self._textures.items():
            self._textures[samplerName] = textureFunction()     # convert the functions to actual textures, since they exist now
        #for i in range(self._numWarp):
        #    levelCount = self.maxLevelCount # very wasteful ... but how do I know if the [input device] will produce mip maps? I need to call [smt].mipLevelCount but what is [smt]?
        #    tex, bufferData, depthMap = dgt.createWarp(self._width,self._height,levelCount=levelCount)
        #    fbos, w, h = bufferData 
        #    self._warpList.append((tex, fbos, depthMap))            # add to our list of warps
        for samplerName, img in self._internalTextureData.items():
            self._textures[samplerName] = dgt.createTexture(img)         # add to our list of textures
        #if warpOnly:
        #    for stack in self._stackList:
        #        for node in stack:
        #            if isinstance(node, Warp):
        #                sceneGraphSet.update(node.setup(width, height))
        #else:
        #for stack in self._stackList:
        #    for node in stack:
        #        sceneGraphSet.update(node.setup(width, height))
        #self._setup = True
        return sceneGraphSet
    def setupGeo(self):
        ''' setup geometry and vbos '''
        if self.shader is None:                                     # make sure our shader is compiled
            self.compileShader()
        self._verts = np.array([1.,1.,0.,  -1.,1.,0.,  1.,-1.,0.,  -1.,-1.,0.,  1.,1.,  0.,1.,  1.,0.,  0.,0.,], dtype=np.float32)        # set up verts and uvs
        self.vertexArray = glGenVertexArrays(1)                                                          # create our vertex array
        glBindVertexArray(self.vertexArray)                                                              # bind our vertex array
        self.vertexBuffer = glGenBuffers(1)                                                              # Generate buffer to hold our vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)                                              # Bind our buffer
        glBufferData(GL_ARRAY_BUFFER, self._verts.nbytes, self._verts, GL_STATIC_DRAW)             # Send the data over to the buffer
        shader_pos = glGetAttribLocation(self.shader, 'position')
        shader_uvs = glGetAttribLocation(self.shader, 'texCoord')
        glEnableVertexAttribArray(shader_pos)                                                            # Add a vertex position attribute
        glVertexAttribPointer(shader_pos, 3, GL_FLOAT, False, 0, None)                                # Describe the position data layout in the buffer
        glEnableVertexAttribArray(shader_uvs)                                                            # Add a vertex uv attribute
        glVertexAttribPointer(shader_uvs, 2, GL_FLOAT, False, 0, ctypes.c_void_p(self._verts[0].nbytes*12))                 # Describe the uv data layout in the buffer
        glBindVertexArray( 0 )                                                                           # Unbind the VAO first (Important)
        glDisableVertexAttribArray(shader_pos)                                                           # Disable our vertex attributes
        glDisableVertexAttribArray(shader_uvs)
        glBindBuffer(GL_ARRAY_BUFFER, 0)                                                              # Unbind the buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)                                                      # Unbind the buffer
    def render(self,resetFBO):
        ''' Render incoming textures' framebuffers then run our shader on our geometry '''
        #print('%s entering render. %s'%(self.__class__, self._name))
        for frameBuffer in self._upstreamBuffers:
            frameBuffer.render(resetFBO)
        if self.shader is None:                                                 # make sure our shader is compiled
            self.compileShader()
        glUseProgram(self.shader)
        for idx, (samplerName, texture) in enumerate(self._textures.items()):
            '''
            data = dgt.readTexture(texture, 0)
            cv2.imshow('%s:%s:%s'%(self._name,samplerName,texture),data)
            cv2.waitKey()
            '''
            #print('%s attaching texture %s to sampler %s on index %s'%(self._name, texture, samplerName, idx))
            dgt.attachTextureNamed(texture, self.shader, idx, samplerName)
        glBindVertexArray(self.vertexArray)                                      # bind our vertex array
        #print('%s executing render. %s'%(self.__class__, self._name))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)                                 # draw a triangle strip
        glBindVertexArray(0)
        glUseProgram(0)
        #print('%s leaving render. %s'%(self.__class__, self._name))
        # should probably return the set of upstream nodes (and thiers) and add ourselves to avoid duplicate rendering in a single frame
        # we could then have a flag for frameRendered or frameComplete that gets checked at beginning of method and reset with new frame
    @property
    def mipLevelCount(self):
        return 1
    @property
    def maxLevelCount(self):
        size = max(self._width, self._height)
        return int(math.floor(math.log(size) / math.log(2.0))) + 1

class Contrast(Warp):
    ''' A shader that will decrease or increase the contrast of an image '''
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.median = 127
        self.factor = 1
        self._fragmentShader = '''
Not implimented yet
'''

class Blur(Warp):
    ''' A specialized 9x9 kernel convolution '''
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # speed up by calculating texCoord in vertex shader and passing them down
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D texRGBA;
// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
	float sStep = 1.0f/512.0f;
	float tStep = 1.0f/512.0f;
    vec4 myColor = texture2D(texRGBA, fragTexCoord + vec2(-sStep, -tStep))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(-sStep, 0))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(-sStep, tStep))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(0, -tStep))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(0, 0))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(0, tStep))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(sStep, -tStep))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(sStep, 0))/9.0f;
	myColor = myColor + texture2D(texRGBA, fragTexCoord + vec2(sStep, tStep))/9.0f;
	FragColor = myColor;
}
'''

class Convolution(Warp):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.kernel = np.matrix([[1.]], dtype=np.float32)
    @property
    def kernel(self):
        return self._kernel
    @kernel.setter
    def kernel(self, value):
        self._kernel = value
        self.shader = None
    @property
    def fragmentShader(self):
        sStep = 1./self._width
        tStep = 1./self._height
        code = []
        code.append('in vec2 fragTexCoord;\n')
        code.append('uniform sampler2D texRGBA; // texRGBA\n')
        code.append('// Force location to 0 to ensure its the first output\n')
        code.append('layout (location = 0) out vec4 FragColor;\n')
        code.append('void main() {\n')
        code.append('    FragColor = vec4(0,0,0,0);\n')
        shape = self._kernel.shape
        #print(shape[0]*shape[1])
        it = np.nditer(self._kernel, flags=['multi_index'])
        while not it.finished:
            if it[0] != 0.:
                s = sStep*(it.multi_index[1]-shape[1]/2.)
                t = tStep*(it.multi_index[0]-shape[0]/2.)
                code.append('    FragColor = FragColor + texture2D(texRGBA, fragTexCoord + vec2(%s, %s))*%sf;\n'%(s, t, it[0]))
            it.iternext()
        code.append('}')
        return '%s%s'%(config.shaderHeader,''.join(code))
    
class Over(Warp):
    ''' A compisiting shader implimentation of the over function '''
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D texOVER_RGBA;  // texOVER_RGBA
uniform sampler2D texUNDER_RGBA; // texUNDER_RGBA

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
    vec4 over = texture2D( texOVER_RGBA, fragTexCoord );
    if (over.a > 0){
        over = vec4(over.rgb/over.a,over.a);
    }
    vec4 under = texture2D( texUNDER_RGBA, fragTexCoord );
    FragColor = mix(under, over, over.a);
}
'''

class Lookup(Warp):
    ''' A lookup table implementation where the lookup table is computed '''
    def __init__(self, name, lutFile, **kwargs):
        super().__init__(name, **kwargs)
        self._lutFile = lutFile
        self._internalTextureData['texLUT'] = dgt.loadImage(self._lutFile)   # load our LUT file
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D texRGBA; // texRGBA
uniform sampler2D texLUT;  // texLUT

layout (location = 0) out vec4 FragColor;

void main() {
    vec2 uv = texture2D(texLUT, fragTexCoord).rg;
    FragColor = texture2D(texRGBA, uv);
};
'''

class Image(Warp):
    ''' A shader displaying the indicated image '''
    def __init__(self, name, imageFile, **kwargs):
        super().__init__(name, **kwargs)
        self._imageFile = imageFile
        self._internalTextureData['texRGBA'] = dgt.loadImage(self._imageFile)   # load our LUT file
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D texRGBA; // texRGBA

layout (location = 0) out vec4 FragColor;

void main() {
    FragColor = texture2D( texRGBA, fragTexCoord );
};
'''


class GaussMIPMap(Warp):
    def __init__(self, name, **kwargs):
        super(GaussMIPMap, self).__init__(name, **kwargs)
    @property
    def fragmentShader(self):        
        with open('GaussMIPTexture2DFragment.glsl', 'r') as fid:
            code = ''.join([line for line in fid])
        return code
    @property
    def mipLevelCount(self):
        return self.maxLevelCount
    def render(self):
        # need to add all the stuff the parent method does - I don't think we can call super, or maybe we can, since it just does level 1, then we do the rest?
        # for all mip levels
        levelRes = np.array([width, height],int)
        for level in range(self.mipLevelCount):
            if level > 0:
                # bind previous outputs as inputs
                for i in range(self._numWarp):
                    dgt.attachTextureNamed(parentTextures, self.shader, len(self._textures) + i, 'tex%dTexture2DFramebufferTexture2D' % i)
                    break

            setUniform(self.shader, "resolution", levelRes)
            setUniform(self.shader, "mipLevelIndex", level)

            glBindFramebuffer(GL_FRAMEBUFFER, (parentFrameBuffers[level] if len(parentFrameBuffers) > 0 else 0))              # Disable our frameBuffer so we can render to screen
            if clear:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glViewport(posWidth, 0, levelRes[0], levelRes[1])                               # set the viewport to the portion we are drawing
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)                                 # draw a triangle strip
            levelRes = np.maximum(levelRes / 2, 1).astype(int)
        glUseProgram(0)
    

class DepthOfField(Warp):
    def __init__(self, name, **kwargs):
        super(DepthOfField, self).__init__(name, **kwargs)
    @property
    def fragmentShader(self):        
        with open('DepthOfFieldPerceptualFragment.glsl', 'r') as fid:
            code = ''.join([line for line in fid])
        return code