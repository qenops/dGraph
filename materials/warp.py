#!/usr/bin/env python
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
from OpenGL.GL import shaders
import ctypes
import dGraph.textures as dgt
import dGraph.materials as dgm
_shaderHeader = dgm._shaderHeader

class Warp(object):
    ''' A warp class that takes an image and alters it in some manner '''
    _warpList = {}                                          # store all existing materials here
    def __new__(cls, name, *args, **kwargs):                # keeps track of all existing materials
        if name in cls._warpList.keys():
            if not isinstance(cls._warpList[name],cls):     # do some type checking to prevent mixed results
                raise TypeError('Material of name "%s" already exists and is type: %s'%(name, type(cls._materialList[name])))
        else:
            cls._warpList[name] = super(Warp, cls).__new__(cls, name, *args, **kwargs)
        return cls._warpList[name]
    def __init__(self, name, **kwargs):      
        self._name = name
        self._setup = False
        self._numWarp = 1
        self._warpList = []     # list of tuples containing (texture, frameBuffer) pair
        self._texImages = []      # list containing texture images
        self._texList = []      # list containing all textures
        self._stackList = []
        self.shader = None
        self._vertexShader = '''
in vec3 position;
in vec2 texCoord;
out vec2 fragTexCoord;
void main() {
    gl_Position = vec4(position,1f);
    fragTexCoord = texCoord;
}
'''
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D tex0;
// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
    FragColor = texture2D( tex0, fragTexCoord );
    //FragColor = vec4(fragTexCoord.x,fragTexCoord.y,1f,1f); 
    //vec4 col = texture2D( tex0, fragTexCoord );
    //FragColor = vec4(col.a, col.a, col.a, col.a);
}
'''
    @property
    def vertexShader(self):
        '''The output of the vertex shader is clip coordinates (not viewport coordinates) - OpenGL still performs the "divide by w" step automatically.'''
        return '%s%s'%(_shaderHeader, self._vertexShader)
    @property
    def fragmentShader(self):
        return '%s%s'%(_shaderHeader,self._fragmentShader)
    def pushRenderStack(self, newRenderStack):
        ''' Push an entire render stack onto our list all at once '''
        self._stackList.append(newRenderStack)
    def compileShader(self):
        self.shader = shaders.compileProgram(
            shaders.compileShader(self.vertexShader, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.fragmentShader, GL.GL_FRAGMENT_SHADER)
        )
    def setup(self, width, height): #, warpOnly=False):
        ''' Setup our geometry and buffers and compile our shaders '''
        if self._setup:
            return set()
        self._width = width
        self._height = height
        self._warpList = []                  # clear textures for resizing
        if not hasattr(self, 'vertexArray'):
            self.setupGeo()
        for i in xrange(self._numWarp):
            tex, bufferData = dgt.createWarp(self._width,self._height)
            frameBuffer, w, h = bufferData 
            self._warpList.append((tex, frameBuffer))            # add to our list of warps
        for img in self._texImages:
            self._texList.append(dgt.createTexture(img))         # add to our list of textures
        sceneGraphSet = set()
        #if warpOnly:
        #    for stack in self._stackList:
        #        for node in stack:
        #            if isinstance(node, Warp):
        #                sceneGraphSet.update(node.setup(width, height))
        #else:
        for stack in self._stackList:
            for node in stack:
                sceneGraphSet.update(node.setup(width, height))
        self._setup = True
        return sceneGraphSet
    def setupGeo(self):
        ''' setup geometry and vbos '''
        if self.shader is None:                                     # make sure our shader is compiled
            self.compileShader()
        self._verts = np.array([1.,1.,0.,  -1.,1.,0.,  1.,-1.,0.,  -1.,-1.,0.,  1.,1.,  0.,1.,  1.,0.,  0.,0.,], dtype=np.float32)        # set up verts and uvs
        self.vertexArray = GL.glGenVertexArrays(1)                                                          # create our vertex array
        GL.glBindVertexArray(self.vertexArray)                                                              # bind our vertex array
        self.vertexBuffer = GL.glGenBuffers(1)                                                              # Generate buffer to hold our vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffer)                                              # Bind our buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._verts.nbytes, self._verts, GL.GL_STATIC_DRAW)             # Send the data over to the buffer
        shader_pos = GL.glGetAttribLocation(self.shader, 'position')
        shader_uvs = GL.glGetAttribLocation(self.shader, 'texCoord')
        GL.glEnableVertexAttribArray(shader_pos)                                                            # Add a vertex position attribute
        GL.glVertexAttribPointer(shader_pos, 3, GL.GL_FLOAT, False, 0, None)                                # Describe the position data layout in the buffer
        GL.glEnableVertexAttribArray(shader_uvs)                                                            # Add a vertex uv attribute
        GL.glVertexAttribPointer(shader_uvs, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(self._verts[0].nbytes*12))                 # Describe the uv data layout in the buffer
        GL.glBindVertexArray( 0 )                                                                           # Unbind the VAO first (Important)
        GL.glDisableVertexAttribArray(shader_pos)                                                           # Disable our vertex attributes
        GL.glDisableVertexAttribArray(shader_uvs)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)                                                              # Unbind the buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)                                                      # Unbind the buffer
    def render(self, width, height, renderStack, parentFrameBuffer=0, posWidth=0, clear=True):
        ''' Recursively render the render stack and render as texture to my geometry '''
        #if width != self._width or height != self._height:
            #print '%sx%s to %sx%s'%(self._width, self._height, width, height)
            #self.setup(width, height, True)
        if self._numWarp > 1 and len(renderStack):
            raise RuntimeError('%s object should be last item in render stack: %s\nPlease use object render stack instead.'%(self.__class__, renderStack))
        if self._numWarp < 2:
            self._stackList = [renderStack]
        for idx in range(self._numWarp):
            stack = list(self._stackList[idx])                                      # get a copy of this texture's render stack
            #stack = self._stackList[idx]
            tex, frameBuffer = self._warpList[idx]                               # get our texture and frameBuffer
            temp = stack.pop()
            temp.render(width, height, stack, frameBuffer, posWidth=0, clear=True)                   # Go up the render stack to get our texture
            #data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)
            #data = np.reshape(data,(height,width,3))
            #cv2.imshow('%s:%s'%(self._name,temp._name),data)
            #cv2.waitKey()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, parentFrameBuffer)              # Disable our frameBuffer so we can render to screen
        if clear:
            #print '%s clearing. %s'%(self.__class__, self._name)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glViewport(posWidth, 0, width, height)                               # set the viewport to the portion we are drawing
        GL.glUseProgram(self.shader)
        for idx in range(self._numWarp):                                        # attach our warp textures first
            tex, frameBuffer = self._warpList[idx] 
            dgt.attachTexture(tex, self.shader, idx)
        for i, tex in enumerate(self._texList):                                 # then attach our normal textures
            idx = i + self._numWarp
            dgt.attachTexture(tex, self.shader, idx)
        GL.glBindVertexArray(self.vertexArray)                                      # bind our vertex array
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)                                 # draw a triangle strip
        GL.glUseProgram(0)
        #print '%s leaving render. %s'%(self.__class__, self._name)

class Contrast(Warp):
    ''' A shader that will decrease or increase the contrast of an image '''
    def __init__(self, name, **kwargs):
        super(Blur, self).__init__(name, **kwargs)
        self.median = 127
        self.factor = 1
        self._fragmentShader = '''
Not implimented yet
'''

class Blur(Warp):
    ''' A specialized 9x9 kernel convolution '''
    def __init__(self, name, **kwargs):
        super(Blur, self).__init__(name, **kwargs)
        # speed up by calculating texCoord in vertex shader and passing them down
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D tex0;
// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
	float sStep = 1f/512f;
	float tStep = 1f/512f;
    vec4 myColor = texture2D(tex0, fragTexCoord + vec2(-sStep, -tStep))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(-sStep, 0))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(-sStep, tStep))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(0, -tStep))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(0, 0))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(0, tStep))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(sStep, -tStep))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(sStep, 0))/9f;
	myColor = myColor + texture2D(tex0, fragTexCoord + vec2(sStep, tStep))/9f;
	FragColor = myColor;
}
'''
class Convolution(Warp):
    def __init__(self, name, **kwargs):
        super(Convolution, self).__init__(name, **kwargs)
        self.kernel = np.matrix([[1.]], dtype=np.float32)
    @property
    def fragmentShader(self):
        sStep = 1./self._width
        tStep = 1./self._height
        code = []
        code.append('in vec2 fragTexCoord;\n')
        code.append('uniform sampler2D tex0;\n')
        code.append('// Force location to 0 to ensure its the first output\n')
        code.append('layout (location = 0) out vec4 FragColor;\n')
        code.append('void main() {\n')
        code.append('    FragColor = vec4(0f,0f,0f,0f);\n')
        shape = self.kernel.shape
        #print(shape[0]*shape[1])
        it = np.nditer(self.kernel, flags=['multi_index'])
        while not it.finished:
            s = sStep*(it.multi_index[1]-shape[1]/2.)
            t = tStep*(it.multi_index[0]-shape[0]/2.)
            code.append('    FragColor = FragColor + texture2D(tex0, fragTexCoord + vec2(%s, %s))*%sf;\n'%(s, t, it[0]))
            it.iternext()
        code.append('}')
        return '%s%s'%(_shaderHeader,''.join(code))
    
class Over(Warp):
    ''' A compisiting shader implimentation of the over function '''
    def __init__(self, name, **kwargs):
        super(Over, self).__init__(name, **kwargs)
        self._numWarp = 2
        for i in range(self._numWarp):
            self._stackList.append([])
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D tex0;  // over
uniform sampler2D tex1;  // under
// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;
void main() {
    vec4 over = texture2D( tex0, fragTexCoord );
    if (over.a > 0){
        over = vec4(over.rgb/over.a,over.a);
    }
    vec4 under = texture2D( tex1, fragTexCoord );
    FragColor = mix(under, over, over.a);
}
'''
    @property
    def overStack(self):
        return list(self._stackList[0])
    @property
    def underStack(self):
        return list(self._stackList[1])
    def overStackAppend(self, value):
        self._stackList[0].append(value)
    def underStackAppend(self, value):
        self._stackList[1].append(value)

class Lookup(Warp):
    ''' A lookup table implementation where the lookup table is computed '''
    def __init__(self, name, lutFile, **kwargs):
        super(Lookup, self).__init__(name, **kwargs)
        self._numWarp = 1
        self._lutFile = lutFile
        self._texImages.append(dgt.loadImage(self._lutFile))   # load our LUT file
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D tex0;  // texCol
uniform sampler2D tex1;  // texLUT

layout (location = 0) out vec4 FragColor;

void main() {
    vec2 uv = texture2D(tex1, fragTexCoord).rg;
    FragColor = texture2D(tex0, uv);
};
'''