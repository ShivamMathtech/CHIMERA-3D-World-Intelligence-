"""Create a synthetic, calibrated RGB-D NPZ and optional MP4 test clip."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from PIL import Image
from chimera.demo import scene

parser=argparse.ArgumentParser()
parser.add_argument('--video',action='store_true',help='Also write a synthetic 8-second test video; requires OpenCV')
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'samples';out.mkdir(exist_ok=True)
rgb,depth,annotations,k=scene(0)
np.savez_compressed(out/'room.npz',rgb=rgb,depth=depth.astype(np.float32),intrinsics=np.array([k[x] for x in ('fx','fy','cx','cy')],dtype=np.float32))
Image.fromarray(rgb).save(out/'room_rgb.png')
print(f'Created {out / "room.npz"} (synthetic camera-space metres)')
if args.video:
    import cv2
    writer=cv2.VideoWriter(str(out/'synthetic-room.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(480,300))
    if not writer.isOpened(): raise RuntimeError('MP4 writer unavailable; install an OpenCV build with video support.')
    try:
        for i in range(96):
            rgb,_,_,_=scene(i/12);writer.write(rgb[:,:,::-1].copy())
    finally: writer.release()
    print('Created synthetic-room.mp4; geometric blocks are not a YOLO accuracy benchmark.')
