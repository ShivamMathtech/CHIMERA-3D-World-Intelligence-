"""Optional real inference adapters. No model installation on application startup."""
from pathlib import Path
import numpy as np
from PIL import Image


class Models:
    def __init__(self):
        self.detector = None
        self.depth_model = None
        self.names = {'detector':'disabled','depth':'disabled'}

    def configure(self, detector_path='', depth_path=''):
        # Load into temporary variables so failed configuration keeps old models intact.
        detector,depth_model = None,None
        if detector_path:
            path = Path(detector_path).expanduser()
            if not path.is_file() or path.suffix.lower() != '.pt':
                raise ValueError('Select a local trusted YOLO .pt checkpoint. See model setup guide.')
            from ultralytics import YOLO
            detector = YOLO(str(path))
        if depth_path:
            path = Path(depth_path).expanduser()
            if not path.is_dir():
                raise ValueError('Depth model must be a local Hugging Face model folder.')
            from transformers import pipeline
            depth_model = pipeline('depth-estimation',model=str(path),device=-1)
        self.detector,self.depth_model = detector,depth_model
        self.names = {'detector':Path(detector_path).name if detector else 'disabled',
                      'depth':Path(depth_path).name if depth_model else 'disabled'}

    def detect(self, rgb, confidence=.35):
        if self.detector is None:
            return []
        result = self.detector.predict(source=rgb[:,:,::-1].copy(),conf=confidence,imgsz=480,device='cpu',verbose=False)[0]
        return [{'box':b.xyxy[0].cpu().tolist(),'label':result.names[int(b.cls[0])],
                 'confidence':float(b.conf[0]),'position':None} for b in result.boxes]

    def depth(self, rgb):
        if self.depth_model is None:
            return None
        prediction = self.depth_model(Image.fromarray(rgb))['predicted_depth']
        raw = prediction.squeeze().cpu().numpy()
        raw = np.asarray(Image.fromarray(raw.astype(np.float32)).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.BILINEAR))
        if not np.isfinite(raw).all():
            raise ValueError('Depth model produced non-finite values.')
        low,high = np.percentile(raw,[2,98])
        # Relative inverse-depth visualisation only; NOT metres or a stable world map.
        inverse = np.clip((raw-low)/max(1e-6,high-low),0,1)
        return (1/(.2+.8*inverse)).astype(np.float32)
