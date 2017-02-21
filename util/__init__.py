#!/usr/bin/env python
'''Mostly a temp storage location for things I need to figure out more permanent homes for

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = []

from OpenGL.GL import *
import dGraph.ui as ui

def draw(corners=[0.,0.,1.,1.]):
    glBegin(GL_QUADS)
    glTexCoord2f(0., 0.)
    glVertex2f(corners[0], corners[1])
    glTexCoord2f(1., 0.)
    glVertex2f(corners[2], corners[1])
    glTexCoord2f(1., 1.)
    glVertex2f(corners[2], corners[3])
    glTexCoord2f(0., 1.)
    glVertex2f(corners[0], corners[3])
    glEnd()

def drawQuad(texture,corners=[0.,0.,1.,1.],size=None):
    width, height = ui.get_framebuffer_size(ui.get_current_context()) if size is None else size
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0., 1., 0., 1., 0., 1.)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    draw(corners)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

def drawWarp(frameTex, frameBuffer, lutTex, imgTex, shader):
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer[0])              # Bind our frame buffer
    drawQuad(imgTex,size=frameBuffer[1:])
    glBindFramebuffer(GL_FRAMEBUFFER, 0)                        # Disable our frameBuffer so we can render to screen
    #glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture
    #glGenerateMipmap(GL_TEXTURE_2D)                             # generate mipmap
    #glBindTexture(GL_TEXTURE_2D, 0)                             # unbind texture
    width, height = ui.get_framebuffer_size(ui.get_current_context())
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glActiveTexture(GL_TEXTURE1)                            # start frameTex: make texture register idx active
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex1")               # get location of our texture
    glUniform1i(texLoc, 1)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE0)                            # start LUTtex: make texture register idx active
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, lutTex)                        # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex0")               # get location of our texture
    glUniform1i(texLoc, 0)                                      # connect location to register idx
    draw()                                                      # draw a quad
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glUseProgram(0)

def drawMixWarp(frameTex, frameBuffer, lut1Tex, lut2Tex, factor, imgTex, shader):
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer[0])           # Bind our frame buffer
    drawQuad(imgTex,size=frameBuffer[1:])
    glBindFramebuffer(GL_FRAMEBUFFER, 0)                        # Disable our frameBuffer so we can render to screen
    width, height = ui.get_framebuffer_size(ui.get_current_context())
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    glActiveTexture(GL_TEXTURE0)                            # start frameTex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, frameTex)                      # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "tex")                # get location of our texture
    glUniform1i(texLoc, 0)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE1)                            # start LUT1tex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, lut1Tex)                       # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "lut1")               # get location of our texture
    glUniform1i(texLoc, 1)                                      # connect location to register idx
    glActiveTexture(GL_TEXTURE2)                            # start LUT2tex: make texture register idx active
    glEnable(GL_TEXTURE_2D)                                     # enable textures
    glBindTexture(GL_TEXTURE_2D, lut2Tex)                       # bind texture to register idx
    texLoc = glGetUniformLocation(shader, "lut2")               # get location of our texture
    glUniform1i(texLoc, 2)                                      # connect location to register idx
    factorLoc = glGetUniformLocation(shader, "factor")      # get the factor location
    glUniform1f(factorLoc,factor)                               # set the factor
    draw()                                                      # draw a quad
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glUseProgram(0)