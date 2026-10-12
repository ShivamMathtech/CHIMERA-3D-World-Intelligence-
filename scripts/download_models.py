"""Explicit, user-invoked model download. Internet and optional AI packages required."""
from pathlib import Path
import argparse
import os


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--detector',action='store_true')
    parser.add_argument('--depth',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    args=parser.parse_args()
    if not (args.detector or args.depth): parser.error('Choose --detector and/or --depth')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.detector:
        from ultralytics import YOLO
        previous=Path.cwd()
        try:
            os.chdir(out)
            YOLO('yolov8n.pt')
        finally: os.chdir(previous)
        print(f'Detector checkpoint: {out / "yolov8n.pt"}')
    if args.depth:
        from huggingface_hub import snapshot_download
        destination=out/'depth-anything-small'
        snapshot_download(repo_id='LiheYoung/depth-anything-small-hf',local_dir=destination,
                          allow_patterns=['*.json','*.safetensors','*.txt'])
        print(f'Depth model directory: {destination}')
    print('Open CHIMERA Settings and paste the absolute paths above. See docs/MODELS.md for licenses.')


if __name__=='__main__':main()
