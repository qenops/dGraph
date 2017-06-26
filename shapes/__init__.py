#!/usr/bin/env python
'''Shape submodule for dGraph scene description module

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Shape", "PolySurface"]

import dGraph as dg
import dGraph.materials as dgm
from dGraph.dio import obj
import dGraph.config as config
from math import sin, cos, pi
import numpy as np
from numpy.linalg import norm
from numpy import dot, cross, matlib
import OpenGL.GL as GL
from OpenGL.GL import shaders
import ctypes

class Shape(dg.WorldObject):
    ''' Any world object that has a renderable surface
        Material (connection to a material class object for illumination information)
        Intersection (given a Ray will return all the intersection points and distances)
        Normal (given a surface point will calculate a normalized normal)
        Bounding Box
        ???? Casts shadows
        ???? U and V values
    '''
    def __init__(self, name, parent):
        super(Shape, self).__init__(name, parent)
        self.classifier = 'shape'
        self.renderable = True

    @property
    def material(self):
        ''' Deprecated '''
        return self._materials[0]
    def setMaterial(self, material):
        ''' Deprecated '''
        self._materials[0] = material

    def generateVBO(self):
        ''' an abstract method for implimentation in subclasses'''
        raise NotImplementedError("Please Implement this method")
    def renderGL(self):
        ''' an abstract method for implimentation in subclasses'''
        raise NotImplementedError("Please Implement this method")
    




class PolySurface(Shape):
    ''' 
        A surface object defined by polygons
        Verticies - local x,y,z and w of each vert
        UVs - uv position list
        Normals - normal vector list 
        Edges - maybe not needed?
        Faces - dict List consisting of verts, uvs, and normals of 3 or more verts that make up a face
        MaterialId - index of material
    '''
    def __init__(self, name, parent, file=None, verts=None, uvs=None, normals=None, faces=None, faceUvs=None, faceNormals=None, faceSizes=None, normalize=False):
        super(PolySurface, self).__init__(name, parent)
        materialIds = None
        materials = None
        if file is not None:
            verts, uvs, normals, faces, faceUvs, faceNormals, faceSizes, materialIds, materials = obj.load(file, normalize)
        self._verts = np.matrix([]) if verts is None else verts
        self._uvs = np.matrix([]) if uvs is None else uvs
        self._normals = np.matrix([]) if normals is None else normals
        self._faceVerts = np.matrix([]) if faces is None else faces
        self._faceUvs = np.matrix([]) if faceUvs is None else faceUvs
        self._faceNormals = np.matrix([]) if faceNormals is None else faceNormals
        self._faceSizes = [] if faceSizes is None else faceSizes
        self._triWorldMatrix = None
        self._materialIds = [] if materialIds is None else materialIds 
        self._materials = [] if materials is None else materials 
        self._VBOdone = False
        self._shader = None

        if len(self._materials) == 0:
            # Add default material
            material = dgm.Material('default')

            self._materialIds = np.matrix(np.zeros([self._faceUvs.shape[0],1], dtype=np.uint64))
            self._materials.append(material)


   
    def triangulateGL(self):
        ''' Generate openGL triangle lists in VVVVVTTTTTNNNNN form
        Don't need to worry about world matrix - we will do that via model matrix '''
        # TODO something if max faceSizes is greater than 3
        if max(self._faceSizes) == 3:
            # Combine all the positions, normals, and uvs into one array, then remove duplicates - that is our vertex buffer
            maxSize = 2**16                                             # numpy uint64 is 64 bits spread over 3 attributes is 21 bits 2**21/3 is max number of faces
            fuvs = np.zeros_like(self._faceVerts, dtype=np.uint64) if len(self._uvs.A1) < 3 else self._faceUvs.astype(np.uint64)
            fnorms = np.zeros_like(self._faceVerts, dtype=np.uint64) if len(self._normals.A1) < 3 else self._faceNormals.astype(np.uint64)
            fmatIds = np.zeros_like(self._faceVerts, dtype=np.uint64) 
            if len(self._materialIds.A1) >= 1: 
                fmatIds = np.resize(self._materialIds.A1, [3,len(self._materialIds.A1)]).transpose().astype(np.uint64) # expand to triangles

            f = np.array(self._faceVerts.astype(np.uint64)+(maxSize*fuvs).astype(np.uint64)+((maxSize**2)*fnorms).astype(np.uint64)+((maxSize**3)*fmatIds).astype(np.uint64)).ravel()
            fullVerts, faces = np.unique(f, return_inverse=True)        # get the unique indices and the reconstruction(our element array)

            # Build our actual vertex array by getting the positions, normals and uvs from our unique indicies
            vertsGL = self._verts[fullVerts%maxSize].getA1()
            uvsGL = np.zeros((0),dtype=np.float32) if len(self._uvs.A1) < 3 else self._uvs[((fullVerts/maxSize)%maxSize).astype(fullVerts.dtype)].getA1()
            normsGL = np.zeros((0),dtype=np.float32) if len(self._normals.A1) < 3 else self._normals[(fullVerts/(maxSize**2)%maxSize).astype(fullVerts.dtype)].getA1()
            matIdsGL = np.zeros((0),dtype=np.float32) if len(self._materialIds.A1) < 1 else (fullVerts/(maxSize**3)).astype(np.float32)
            
            return np.concatenate((vertsGL,uvsGL,normsGL,matIdsGL)), faces.astype(np.uint32), [len(vertsGL),len(uvsGL),len(normsGL),len(matIdsGL)]

    def generateVBO(self):
        ''' generates OpenGL VBO and VAO objects '''
        if self._VBOdone:
            return
        #global shader_pos, shader_uvs, shader_norm
        vertsGL, facesGL, lengths = self.triangulateGL()                                                    # make sure our vert list and face list are populated
        self.numTris = len(facesGL)
        if self._shader is None:                                                                   # make sure our shader is compiled
            self.compileShader()
        self.vertexArray = GL.glGenVertexArrays(1)                                                          # create our vertex array
        GL.glBindVertexArray(self.vertexArray)                                                              # bind our vertex array
        self.vertexBuffer = GL.glGenBuffers(1)                                                              # Generate buffer to hold our vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffer)                                              # Bind our buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertsGL.nbytes, vertsGL, GL.GL_STATIC_DRAW)                     # Send the data over to the buffer
        # Need to figure out stride and normal offset before we start
        stride = 0                                                                                          # stride does not work
        #stride = facesGL[0].nbytes
        #normOffset = self._verts[0].nbytes*lengths[0]+self._verts[0].nbytes*lengths[1]                      # offset will be length of positions + length of uvs
        
        # Set up the array attributes
        shader_pos = GL.glGetAttribLocation(self._shader, 'position')
        shader_uvs = GL.glGetAttribLocation(self._shader, 'texCoord')                              # will return -1 if attribute isn't supported in shader
        shader_norm = GL.glGetAttribLocation(self._shader, 'normal')                               # will return -1 if attribute isn't supported in shader
        shader_materialId = GL.glGetAttribLocation(self._shader, 'materialId')                               # will return -1 if attribute isn't supported in shader
        
        GL.glEnableVertexAttribArray(shader_pos)                                                            # Add a vertex position attribute
        GL.glVertexAttribPointer(shader_pos, 3, GL.GL_FLOAT, False, stride, None)                           # Describe the position data layout in the buffer
              
        if len(self._uvs.A1) > 2 and shader_uvs != -1:
            GL.glEnableVertexAttribArray(shader_uvs)                                                           # Add a vertex uv attribute
            GL.glVertexAttribPointer(shader_uvs, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:1]))*facesGL[0].nbytes))

        if len(self._normals.A1) > 2 and shader_norm != -1:
            GL.glEnableVertexAttribArray(shader_norm)                                                           # Add a vertex uv attribute
            GL.glVertexAttribPointer(shader_norm, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:2]))*facesGL[0].nbytes))

        if len(self._materialIds.A1) > 2 and shader_materialId != -1:
            GL.glEnableVertexAttribArray(shader_materialId)                                                           # Add a vertex material attribute
            GL.glVertexAttribPointer(shader_materialId, 1, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:3]))*facesGL[0].nbytes))
        #import pdb;pdb.set_trace()

        # Create face element array
        self.triangleBuffer = GL.glGenBuffers(1)                                                            # Generate buffer to hold our face data
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.triangleBuffer)                                    # Bind our buffer as element array
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, facesGL.nbytes, facesGL, GL.GL_STATIC_DRAW)             # Send the data over to the buffer
        GL.glBindVertexArray( 0 )                                                                           # Unbind the VAO first (Important)
        GL.glDisableVertexAttribArray(shader_pos)                                                           # Disable our vertex attributes
        GL.glDisableVertexAttribArray(shader_uvs) if len(self._uvs.A1) > 2 and shader_uvs != -1 else True
        GL.glDisableVertexAttribArray(shader_norm) if len(self._normals.A1) > 2 and shader_norm != -1 else True
        GL.glDisableVertexAttribArray(shader_materialId) if len(self._materialIds.A1) > 2 and shader_materialId != -1 else True
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)                                                              # Unbind the buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)                                                      # Unbind the buffer
        self._VBOdone = True

    def renderGL(self, filmMatrix=None, cameraMatrix=None):
        GL.glUseProgram(self._shader)
        # multiply the world transform to the camera matrix
        cameraMatrix = matlib.identity(4) if cameraMatrix is None else cameraMatrix
        filmMatrix = matlib.identity(4) if filmMatrix is None else filmMatrix

        modelMatrix = np.ascontiguousarray(self.worldMatrix.value, dtype=np.float32)
        location = GL.glGetUniformLocation(self._shader, 'modelMatrix')
        GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, modelMatrix)

        viewMatrix = np.ascontiguousarray(cameraMatrix.value, dtype=np.float32)
        location = GL.glGetUniformLocation(self._shader, 'viewMatrix')
        GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, viewMatrix)

        projectionMatrix = np.ascontiguousarray(filmMatrix, dtype=np.float32)
        location = GL.glGetUniformLocation(self._shader, 'projectionMatrix')
        GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, projectionMatrix)

        samplerCount = 0
        for (i,material) in enumerate(self._materials):
            matName = 'material%02d' % i
            samplerCount += material.pushToShader(matName, self._shader, samplerCount)

        # bind our VAO and draw it
        GL.glBindVertexArray(self.vertexArray)
        GL.glDrawElements(GL.GL_TRIANGLES,self.numTris,GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        
    @property
    def vertexShader(self): 
        return config.shaderHeader + """
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;
in vec2 texCoord;
in float materialId;

out vec3 fragNormal;
out vec4 fragPosition;
out vec2 fragTexCoord;
flat out float fragMaterialId;

void main()
{
    mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
    gl_Position = mvp * vec4(position, 1.0);
    fragNormal = normalize(transpose(inverse(modelMatrix)) * vec4(normal,1)).xyz;
    fragPosition = viewMatrix * modelMatrix * vec4(position, 1.0);
    fragTexCoord = texCoord;
    fragMaterialId = materialId;
}
            """

    @property
    def fragmentShader(self): 
        code = config.shaderHeader;
        code += '''

uniform mat4 viewMatrix;

smooth in vec3 fragNormal;      // normal in camera space
smooth in vec4 fragPosition;    // position in camera space
smooth in vec2 fragTexCoord;
flat in float fragMaterialId;

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;

'''
        code += dgm.Material.getShaderStruct()
        for (i,material) in enumerate(self._materials):
            matName = 'material%02d' % i
            code += material.fragmentShaderDefinitions(matName)
            code += material.fragmentShaderShading(matName)

        #code += '''
        #void main() {
        #    FragColor = vec4(0);
        #}
        #'''
        #return code

        code += '''

void main() {
    vec3 lightPosition = vec3(2.0f,3.0f,4.0f);
    vec3 ligthColor = vec3(2.0f);

    vec3 normal = normalize(fragNormal);
    vec3 viewerPos = (inverse(viewMatrix) * vec4(0, 0, 0, 1)).xyz;

    vec3 outDirection = normalize(viewerPos - fragPosition.xyz);
    vec3 inDirection = normalize(lightPosition - fragPosition.xyz);   

    vec3 color = vec3(0);
'''
        for (i,material) in enumerate(self._materials):
            matName = 'material%02d' % i
            code += '''
    if (int(fragMaterialId + 0.5) == {materialId}) {{
        color += {name}_shading(inDirection, outDirection, normal, fragTexCoord);
    }}
            '''.format(name = matName, materialId = i)
        code += '''
    color *= ligthColor;
    
    FragColor.rgb = color;
    //FragColor.rgb = normal;
    FragColor.a = 1;

    //FragColor.rgb = vec3(0);
    //FragColor.rg = fragTexCoord;//material00.diffuseColor;
}'''        
        return code;

    def compileShader(self):
        #print(self.fragmentShader)
        #print(self.fragmentShader.split('\n')[37])
        #import pdb; pdb.set_trace()
        self._shader = shaders.compileProgram(
            shaders.compileShader(self.vertexShader, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.fragmentShader, GL.GL_FRAGMENT_SHADER)
        )