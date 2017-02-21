#!/usr/bin/env python
'''A library of shader classes for rendering surfaces

David Dunn
Sept 2014 - Created
July 2015 - Added openGL support

ALL UNITS ARE IN METERS 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.5'

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import shaders
import ctypes
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

class Warp(object):
    ''' A warp class that takes an image and alters it in some manner '''
    _warpList = {}                                          # store all existing materials here
    def __new__(cls, name, *args, **kwargs):                    # keeps track of all existing materials
        if name in cls._warpList.keys():
            if not isinstance(cls._warpList[name],cls):     # do some type checking to prevent mixed results
                raise TypeError('Material of name "%s" already exists and is type: %s'%(name, type(cls._materialList[name])))
        else:
            cls._warpList[name] = super(Warp, cls).__new__(cls, name, *args, **kwargs)
        return cls._warpList[name]
    def __init__(self, name, **kwargs):      
        self._name = name
        self._numTex = 1
        self._textureList = []      # list of tuples containing (texture, frameBuffer) pair
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
        self._width = width
        self._height = height
        self._textureList = []                  # clear textures for resizing
        if not hasattr(self, 'vertexArray'):
            self.setupGeo()
        for i in xrange(self._numTex):
            tex = GL.glGenTextures(1)                              # setup our texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)                # bind texture
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT);
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_MIRRORED_REPEAT);
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)   # Allocate memory
            # setup frame buffer
            frameBuffer = GL.glGenFramebuffers(1)                                                                  # Create frame buffer
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, frameBuffer)                                                   # Bind our frame buffer
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)        # Attach texture to frame buffer
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self._textureList.append((tex, frameBuffer))                                                           # add to our list of textures
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
        if self._numTex > 1 and len(renderStack):
            raise RuntimeError('%s object should be last item in render stack: %s\nPlease use object render stack instead.'%(self.__class__, renderStack))
        if self._numTex < 2:
            self._stackList = [renderStack]
        for idx in range(self._numTex):
            stack = list(self._stackList[idx])                                      # get a copy of this texture's render stack
            #stack = self._stackList[idx]
            tex, frameBuffer = self._textureList[idx]                               # get our texture and frameBuffer
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
        for idx in range(self._numTex):
            tex, frameBuffer = self._textureList[idx] 
            GL.glActiveTexture(getattr(GL,'GL_TEXTURE%s'%idx))                      # make texture register idx active
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)                                 # bind texture to register idx
            texLoc = GL.glGetUniformLocation(self.shader, "tex%s"%idx)              # get location of our texture
            GL.glUniform1i(texLoc, idx)                                             # connect location to register idx
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
        print(shape[0]*shape[1])
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
        self._numTex = 2
        for i in range(self._numTex):
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
        self._numTex = 2
        self._lutFile = lutFile
        self._fragmentShader = '''
in vec2 fragTexCoord;
uniform sampler2D tex0;  // texCol
uniform sampler2D tex1;  // texLUT

layout (location = 0) out vec4 FragColor;

void main() {
    vec4 uvraw = texture2D(tex1, fragTexCoord);   
    vec2 uv = vec2(uvraw.r/256.0,uvraw.g/256.0);
    FragColor = texture2D(tex0, uv.xy);
};
'''
    def setup(self, width, height): #, warpOnly=False):
        ''' Setup our geometry and buffers and compile our shaders '''
        self._width = width
        self._height = height
        self._textureList = []                  # clear textures for resizing
        if not hasattr(self, 'vertexArray'):
            self.setupGeo()
        # setup our color texture
        tex = GL.glGenTextures(1)                             
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)                # bind texture
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT);
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_MIRRORED_REPEAT);
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)   # Allocate memory
        # setup frame buffer
        frameBuffer = GL.glGenFramebuffers(1)                                                                  # Create frame buffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, frameBuffer)                                                   # Bind our frame buffer
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)        # Attach texture to frame buffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._textureList.append((tex, frameBuffer))                                                           # add to our list of textures
        # load our LUT file
        #image = cv2.imread(self._lutFile, cv2.CV_LOAD_IMAGE_COLOR)
        # setup our LUT texture
        tex = GL.glGenTextures(1)                             
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)                # bind texture
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT);
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_MIRRORED_REPEAT);
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image)   # Allocate and put our image there
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
        return sceneGraphSet