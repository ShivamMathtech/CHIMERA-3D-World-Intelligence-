"""Deterministic synthetic RGB-D room: geometry is ray-cast, never inferred AI."""
import numpy as np
from PIL import Image, ImageDraw

W,H = 480,300
K = {'fx':370.,'fy':370.,'cx':W/2,'cy':H/2}
COLORS = {'person':[34,169,255],'chair':[0,218,154],'table':[255,173,65],
          'sofa':[84,137,255],'plant':[40,210,145],'laptop':[186,107,255]}


def scene(t):
    # Bounds in camera coordinates, camera at the origin looking along +z.
    p = -1.05 + .65*np.sin(t*.22)
    boxes = [
        ('sofa',[-1.7,-1.05,3.8],[-.2,-.25,4.55]),
        ('table',[-.55,-1.1,2.6],[.55,-.66,3.4]),
        ('chair',[-1.65,-1.1,2.1],[-1.12,-.25,2.65]),
        ('chair',[.85,-1.1,3.3],[1.35,-.2,3.85]),
        ('laptop',[-.2,-.65,2.8],[.25,-.40,3.02]),
        ('plant',[1.6,-1.1,4.1],[2.05,.35,4.55]),
        ('person',[p,-1.1,4.85],[p+.35,.55,5.15]),
        ('person',[1.+.22*np.sin(t*.13),-1.1,2.3],[1.35+.22*np.sin(t*.13),.55,2.6]),
    ]
    v,u = np.mgrid[:H,:W]
    directions = np.stack(((u-K['cx'])/K['fx'], -(v-K['cy'])/K['fy'],np.ones_like(u)),axis=-1)
    depth = np.full((H,W),6.0)
    rgb = np.zeros((H,W,3),dtype=np.uint8); rgb[:] = [43,64,80]
    # Room surfaces: back wall, floor and side walls.
    surfaces = [(1,-1.1,[51,70,79]),(0,-2.3,[40,60,73]),(0,2.3,[53,67,79]),(1,1.3,[36,52,67])]
    for axis,value,color in surfaces:
        with np.errstate(divide='ignore',invalid='ignore'):
            distance = value/directions[:,:,axis]
        valid = (distance > 0) & (distance < depth)
        depth[valid] = distance[valid]; rgb[valid] = color
    # Floor tiles and back-wall window enhance spatial reference without photorealism claims.
    world = directions*depth[:,:,None]
    grid = (np.abs((world[:,:,0]*3)%1-.5)>.47) | (np.abs((world[:,:,2]*3)%1-.5)>.47)
    rgb[grid] = np.minimum(rgb[grid]+12,255)
    window = (world[:,:,2]>5.99)&(world[:,:,0]<-.6)&(world[:,:,0]>-1.7)&(world[:,:,1]>.0)&(world[:,:,1]<.9)
    rgb[window] = [115,175,194]
    detections = []
    inv = np.divide(1,directions,out=np.full_like(directions,1e8),where=np.abs(directions)>1e-8)
    for label,lo,hi in boxes:
        t1,t2 = np.array(lo)*inv,np.array(hi)*inv
        enter = np.minimum(t1,t2).max(axis=2)
        leave = np.maximum(t1,t2).min(axis=2)
        visible = (enter>0)&(leave>=enter)&(enter<depth)
        depth[visible] = enter[visible]
        rgb[visible] = COLORS[label]
        if visible.any():
            ys,xs = np.where(visible)
            detections.append({'label':label,'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'confidence':1.,'position':None})
    return rgb,depth,detections,K.copy()
