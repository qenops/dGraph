#!/usr/bin/env python
'''GLUT interface submodule for dGraph scene description module

David Dunn
Feb 2017 - created

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = []

import OpenGL.GL as GL
from OpenGL.GL import *
import OpenGL.GLUT as GLUT

def init():
    GLUT.glutInit()
    #GLUT.glutInitDisplayMode(GLUT.GLUT_RGB | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
    GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
    GLUT.glutInitWindowSize(width, height)
    # the window starts at the upper left corner of the screen 
    GLUT.glutInitWindowPosition(0, 0)
    window = GLUT.glutCreateWindow(winName)
    GLUT.glutDisplayFunc(DrawGLScene)
    # Uncomment this line to get full screen.
    #glutFullScreen()
    # When we are doing nothing, redraw the scene.
    GLUT.glutIdleFunc(DrawGLScene)
    GLUT.glutReshapeFunc(resizeWindow)
    GLUT.glutKeyboardFunc(keyPressed)
    GLUT.glutSpecialFunc(keyPressed)

def mainLoop():
    # Start Event Processing Engine 
    GLUT.glutMainLoop()

def setWindowTitle(text):
    GLUT.glutSetWindowTitle(text)

# The function called whenever a key is pressed. Tuples to pass in: (key, x, y)  
def keyPressed(*args):
    ''' Directory of key presses:
    Camera Motions:
        Translate:  w = up
                    s = down
                    a = left
                    d = right
        Rotate:     i = -z
                    k = +z
                    j = -x
                    l = +x
        IPD:        z = increase IPD
                    x = decrease IPD
        Output:     p = Print camera info
    Deconvolution:
        Noise to Signal Ratio:  [ = decrease nsr
                                ] = increase nsr
        Aperture size:          - = smaller
                                = = bigger
    Write Images (three buffers for each):
        Standard Blur (What you see is what you get):   [0-3]
        Deconvolution Blur:                             [4-6]
        Band Stop Filter:                               [7-9]
    '''
    global cameras, objects, nsr, aperture
    translate = np.array((0.,0.,0.))
    rotate = np.array((0.,0.,0.))
    ipd = 0
    ESCAPE = '\033'
    # If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit ()
    if args[0] == 'w':
        translate -= np.array((0.,0.,0.1))
    if args[0] == 's':
        translate += np.array((0.,0.,0.1))
    if args[0] == 'a':
        translate -= np.array((0.005,0.,0.))
    if args[0] == 'd':
        translate += np.array((0.005,0.,0.))
    if args[0] == 'i':
        rotate -= np.array((0.,5.,0.))
    if args[0] == 'k':
        rotate += np.array((0.,5.,0.))
    if args[0] == 'j':
        rotate -= np.array((5.,0.,0.))
    if args[0] == 'l':
        rotate += np.array((5.,0.,0.))
    if args[0] == 't':
        objects['teapot'].translate -= np.array((0.,0.,0.1))
    if args[0] == 'g':
        objects['teapot'].translate += np.array((0.,0.,0.1))
    if args[0] == 'z':
        ipd += 0.005
    if args[0] == 'x':
        ipd -= 0.005
    if args[0] == '[':
        nsr *= 0.75
    if args[0] == ']':
        nsr *= 4/3.
    if args[0] == '-':
        aperture -= 0.001
    if args[0] == '=':
        aperture += 0.001
    if args[0] == 'p':
        print('Translate = %s'%cameras[0].translate)
        print('Rotate = %s'%cameras[0].rotate)
        print('IPD = %s'%cameras[0].IPD)
        print('nsr = %s'%nsr)
        print('aperture = %s'%aperture)
    for cam in cameras:
        cam.IPD += ipd
    for name, obj in objects.items():
        obj.translate += translate
        obj.rotate += rotate
    if isinstance(args[0], basestring) and args[0].isdigit():
        writeImages(int(args[0]))
    return


def drawText( value, x,y,  windowHeight, windowWidth, color=(1,1,1), font = GLUT.GLUT_BITMAP_HELVETICA_18, step = 18 ):
    """Draw the given text at given 2D position in window
       http://www.math.uiuc.edu/~gfrancis/illimath/windows/aszgard_mini/pylibs/OpenGLContext/tests/glutbitmapcharacter.py
    """
    glMatrixMode(GL_PROJECTION);
    # For some reason the GL_PROJECTION_MATRIX is overflowing with a single push!
    # glPushMatrix()
    matrix = glGetDouble( GL_PROJECTION_MATRIX )

    glDisable(GL_DEPTH_TEST)
    glLoadIdentity();
    glOrtho(0.0, windowHeight or 32, 0.0, windowWidth or 32, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRasterPos2i(int(x), int(y));
    glColor3f(color[0], color[1], color[2])
    lines = 0
    for character in value:
        if character == '\n':
            glRasterPos2i(int(x), int(y)-(lines*step))
        else:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord(character));
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    # For some reason the GL_PROJECTION_MATRIX is overflowing with a single push!
    # glPopMatrix();
    glLoadMatrixd( matrix ) # should have un-decorated alias for this...

    glMatrixMode(GL_MODELVIEW);