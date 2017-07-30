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
    def __init__(self, name, parent, file=None, verts=None, uvs=None, normals=None, faces=None, faceUvs=None, faceNormals=None, faceSizes=None, normalize=False, scene=None):
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

        self._tangents = np.matrix([])
        self._bitangents = np.matrix([])

        if len(self._materials) == 0:
            # Add default material
            material = dgm.Material('default')

            self._materialIds = np.matrix(np.zeros([self._faceUvs.shape[0],1], dtype=np.uint64))
            self._materials.append(material)

        if scene is None and isinstance(parent, dg.SceneGraph):
            scene = parent
        if scene is None:
            raise RuntimeError('We really do need scene...')
        self._scene = scene

   
    def triangulateGL(self):
        ''' Compute tangents and bitangents'''
        # http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
        if len(self._faceUvs.A1) > 0:
            self._tangents = np.zeros(self._normals.shape, self._normals.dtype)
            self._bitangents = np.zeros(self._normals.shape, self._normals.dtype)

            for triIndex in range(self._faceVerts.shape[0]):

                # Shortcuts for vertices
                v0 = self._verts[self._faceVerts[triIndex,0]].A1;
                v1 = self._verts[self._faceVerts[triIndex,1]].A1;
                v2 = self._verts[self._faceVerts[triIndex,2]].A1;

                # Shortcuts for UVs
                uv0 = self._uvs[self._faceUvs[triIndex,0]].A1;
                uv1 = self._uvs[self._faceUvs[triIndex,1]].A1;
                uv2 = self._uvs[self._faceUvs[triIndex,2]].A1;

                # Edges of the triangle : postion delta
                deltaPos1 = v1-v0;
                deltaPos2 = v2-v0;

                # UV delta
                deltaUV1 = uv1-uv0;
                deltaUV2 = uv2-uv0;

                denom = (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0])
                denom = max(denom, 1e-5) if denom >= 0 else min(denom, -1e-5)
                r = 1.0 / denom;
                tangent = (deltaPos1 * deltaUV2[1]   - deltaPos2 * deltaUV1[1])*r;
                bitangent = (deltaPos2 * deltaUV1[0]   - deltaPos1 * deltaUV2[0])*r;
                
                for i in range(3):
                    self._tangents[self._faceNormals[triIndex,i]] += tangent 
                    self._bitangents[self._faceNormals[triIndex,i]] += bitangent

            for i in range(len(self._tangents)):
                self._tangents[i] /= np.linalg.norm(self._tangents[i], 2)
                self._bitangents[i] /= np.linalg.norm(self._bitangents[i], 2)

                # orthogonalize
                self._tangents[i] = (self._tangents[i] - np.dot(self._tangents[i], self._normals[i].A1) * self._normals[i].A1);
                self._bitangents[i] = np.cross(self._normals[i].A1, self._tangents[i]);

            self._tangents = np.matrix(self._tangents)
            self._bitangents = np.matrix(self._bitangents)
          

        ''' Generate openGL triangle lists in VVVVVTTTTTNNNNN form
        Don't need to worry about world matrix - we will do that via model matrix '''
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
        tangentsGL = np.zeros((0),dtype=np.float32) if len(self._tangents.A1) < 3 else self._tangents[(fullVerts/(maxSize**2)%maxSize).astype(fullVerts.dtype)].getA1() 
        bitangentsGL = np.zeros((0),dtype=np.float32) if len(self._bitangents.A1) < 3 else self._bitangents[(fullVerts/(maxSize**2)%maxSize).astype(fullVerts.dtype)].getA1()
        matIdsGL = np.zeros((0),dtype=np.float32) if len(self._materialIds.A1) < 1 else (fullVerts/(maxSize**3)).astype(np.float32)
            
        
        return np.concatenate((vertsGL,uvsGL,normsGL,tangentsGL,bitangentsGL,matIdsGL)), \
                faces.astype(np.uint32), \
                [len(vertsGL),len(uvsGL),len(normsGL),len(tangentsGL),len(bitangentsGL),len(matIdsGL)]

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
        shader_tangent = GL.glGetAttribLocation(self._shader, 'tangent')                               # will return -1 if attribute isn't supported in shader
        shader_bitangent = GL.glGetAttribLocation(self._shader, 'bitangent')                               # will return -1 if attribute isn't supported in shader
        shader_materialId = GL.glGetAttribLocation(self._shader, 'materialId')                               # will return -1 if attribute isn't supported in shader
        
        GL.glEnableVertexAttribArray(shader_pos)                                                            # Add a vertex position attribute
        GL.glVertexAttribPointer(shader_pos, 3, GL.GL_FLOAT, False, stride, None)                           # Describe the position data layout in the buffer
              
        if len(self._uvs.A1) > 2 and shader_uvs != -1:
            GL.glEnableVertexAttribArray(shader_uvs)                                                           # Add a vertex uv attribute
            GL.glVertexAttribPointer(shader_uvs, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:1]))*facesGL[0].nbytes))

        if len(self._normals.A1) > 2 and shader_norm != -1:
            GL.glEnableVertexAttribArray(shader_norm)                                                           
            GL.glVertexAttribPointer(shader_norm, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:2]))*facesGL[0].nbytes))

        if len(self._tangents.A1) > 2 and shader_tangent != -1:
            GL.glEnableVertexAttribArray(shader_tangent)                                                           
            GL.glVertexAttribPointer(shader_tangent, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:3]))*facesGL[0].nbytes))

        if len(self._bitangents) > 2 and shader_bitangent != -1:
            GL.glEnableVertexAttribArray(shader_bitangent)                                                           
            GL.glVertexAttribPointer(shader_bitangent, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:4]))*facesGL[0].nbytes))

        if len(self._materialIds.A1) > 2 and shader_materialId != -1:
            GL.glEnableVertexAttribArray(shader_materialId)                                                           # Add a vertex material attribute
            GL.glVertexAttribPointer(shader_materialId, 1, GL.GL_FLOAT, False, stride, ctypes.c_void_p(int(np.sum(lengths[:5]))*facesGL[0].nbytes))
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
        self._scene.pushLightsToShader(self._shader)

        GL.glDisable(GL.GL_CULL_FACE)

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
in vec3 tangent;
in vec3 bitangent;
in vec2 texCoord;
in float materialId;

out vec3 fragNormal;
out vec3 fragTangent;
out vec3 fragBitangent;
out vec4 fragPosition;
out vec2 fragTexCoord;
flat out float fragMaterialId;

void main()
{
    mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
    gl_Position = mvp * vec4(position, 1.0);

    mat4 normalMatrix = transpose(inverse(viewMatrix * modelMatrix));
    fragNormal = normalize(normalMatrix * vec4(normal, 0.0)).xyz;
    fragTangent = normalize(normalMatrix * vec4(tangent, 0.0)).xyz;
    fragBitangent = normalize(normalMatrix * vec4(bitangent, 0.0)).xyz;
    if (length(tangent) <= 1e-3) {
        fragTangent = vec3(0);
        fragBitangent = vec3(0);
    }

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
smooth in vec3 fragTangent;      // tangent in camera space
smooth in vec3 fragBitangent;      // bitangent in camera space
smooth in vec4 fragPosition;    // position in camera space
smooth in vec2 fragTexCoord;
flat in float fragMaterialId;

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;

'''
        code += self._scene.fragmentShaderLights()
        code += dgm.Material.getShaderStruct()
        for (i,material) in enumerate(self._materials):
            matName = 'material%02d' % i
            code += material.fragmentShaderDefinitions(matName)
            code += material.fragmentShaderShading(matName, self._scene)

        #code += '''
        #void main() {
        #    FragColor = vec4(0);
        #}
        #'''
        #return code


        code += '''

void main() {
    
    vec3 viewerPos = (inverse(viewMatrix) * vec4(0, 0, 0, 1)).xyz;

    vec3 outDirection = normalize(viewerPos - fragPosition.xyz);

    vec3 N = normalize(fragNormal);
    vec3 T = vec3(0);
    vec3 B = vec3(0);
    if (length(fragTangent) > 1e-3) {
        T = normalize(fragTangent);
        B = normalize(fragBitangent);

        T = normalize(T - dot(T, N) * N);
        B = cross(N, T);
    }

    vec3 color = vec3(0);
'''
        for (i,material) in enumerate(self._materials):
            matName = 'material%02d' % i
            code += '''
    if (int(fragMaterialId + 0.5) == {materialId}) {{
        color += {name}_shading(fragPosition.xyz, outDirection, N, T, B, fragTexCoord);
    }}
            '''.format(name = matName, materialId = i)
        code += '''
    
    FragColor.rgb = color;
    FragColor.a = 1;

    //FragColor.rgb = vec3(0);
    //FragColor.rg = fragTexCoord;//material00.diffuseColor;
}'''        
        #print(code);
        return code;

    def compileShader(self):
        #print(self.fragmentShader)
        #print(self.fragmentShader.split('\n')[37])
        #import pdb; pdb.set_trace()
        self._shader = shaders.compileProgram(
            shaders.compileShader(self.vertexShader, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.fragmentShader, GL.GL_FRAGMENT_SHADER)
        )