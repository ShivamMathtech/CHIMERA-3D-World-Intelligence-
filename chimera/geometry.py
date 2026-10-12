"""Camera-space reconstruction. Coordinates: x right, y up, z forward."""
import numpy as np


def reconstruct(rgb, depth, intrinsics, step=5, max_depth=20.0):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError('Aligned RGB and depth must have identical width and height.')
    fx, fy, cx, cy = [float(intrinsics[k]) for k in ('fx','fy','cx','cy')]
    if not all(np.isfinite([fx,fy,cx,cy])) or fx <= 0 or fy <= 0:
        raise ValueError('Intrinsics must be finite; focal lengths must be positive.')
    if not 1 <= step <= 32:
        raise ValueError('Point stride must be 1–32.')
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    zz = depth[::step, ::step]
    valid = np.isfinite(zz) & (zz > .05) & (zz < max_depth)
    safe_z = np.where(valid, zz, 0)
    xyz = np.stack(((uu-cx)*safe_z/fx, -(vv-cy)*safe_z/fy, safe_z), axis=-1)
    points = np.concatenate((xyz[valid], rgb[::step, ::step, :3][valid]), axis=1)
    ids = np.full(zz.shape, -1, dtype=int)
    ids[valid] = np.arange(valid.sum())
    # Reject edges across depth discontinuities instead of bridging foreground/background.
    faces = []
    for a,b,c in ((ids[:-1,:-1],ids[1:,:-1],ids[:-1,1:]),
                  (ids[1:,1:],ids[:-1,1:],ids[1:,:-1])):
        triangles = np.stack((a,b,c),axis=-1).reshape(-1,3)
        triangles = triangles[(triangles >= 0).all(axis=1)]
        if len(triangles):
            z = points[triangles,2]
            triangles = triangles[np.ptp(z,axis=1) < np.maximum(.10, z.mean(axis=1)*.08)]
            faces.extend(triangles.tolist())
    return points, faces, float(valid.mean())


def locate(box, depth, intrinsics):
    if depth is None:
        return None
    x1,y1,x2,y2 = box
    h,w = depth.shape
    u,v = int((x1+x2)/2),int((y1+y2)/2)
    patch = depth[max(0,v-3):min(h,v+4),max(0,u-3):min(w,u+4)]
    good = patch[np.isfinite(patch) & (patch > .05) & (patch < 20)]
    if not good.size:
        return None
    z = float(np.median(good))
    return [(u-intrinsics['cx'])*z/intrinsics['fx'], -(v-intrinsics['cy'])*z/intrinsics['fy'], z]


def ply_bytes(points, faces=None):
    faces = faces or []
    header = 'ply\nformat ascii 1.0\ncomment CHIMERA camera coordinates x right y up z forward\n'
    header += f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
    header += f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'
    rows = ['%.5f %.5f %.5f %d %d %d' % tuple(p) for p in points]
    rows += ['3 %d %d %d' % tuple(f) for f in faces]
    return (header+'\n'.join(rows)+'\n').encode()
