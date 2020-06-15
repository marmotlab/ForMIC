#helps us make pretty pictures :)

import math
import numpy as np
from gym.envs.classic_control import rendering


def create_circle(x, y, diameter, box_size, fill, resolution=20):
    box_center = ((x + box_size / 2),(y + box_size / 2))

    thetas = np.linspace(0, 2.0 * math.pi, resolution)
    xs = box_center[0] + np.cos(thetas) * diameter/2
    ys = box_center[1] + np.sin(thetas) * diameter/2
        
    circ=rendering.FilledPolygon(list(zip(xs,ys)))
    circ.set_color(fill[0],fill[1],fill[2])
    circ.add_attr(rendering.Transform())
    return circ

def create_rectangle(x,y,width,height,color,hollow=False):
    ps=[(x,y),((x+width),y),((x+width),(y+height)),(x,(y+height))]
    if hollow:
        rect = rendering.PolyLine(ps, True)
    else:
        rect = rendering.FilledPolygon(ps)
    rect.set_color(color[0],color[1],color[2])
    rect.add_attr(rendering.Transform())
    return rect

#d = diameter, h = height
def create_semifilled_circle(x, y, d, box_size, fill1, fill2, h, resolution=20):
    box_center = ((x + box_size / 2),(y + box_size / 2))

    if h <= 0.5:
        theta = np.arccos(1 - 2 * h)
        thetaPrime = np.pi / 2 - theta
        thetas = np.linspace(-thetaPrime, -(thetaPrime + 2 * theta), resolution)
        
    else:
        thetaPrime = np.arcsin(2 * h - 1)
        thetas = np.linspace(-(thetaPrime + np.pi), thetaPrime, resolution)

    xs = box_center[0] + np.cos(thetas) * d / 2
    ys = box_center[1] + np.sin(thetas) * d / 2
        
    circ=rendering.FilledPolygon(list(zip(xs,ys)))
    circ.set_color(fill1[0],fill1[1],fill1[2])
    circ.add_attr(rendering.Transform())
    return circ
