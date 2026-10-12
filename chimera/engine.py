import io
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from . import demo
from .geometry import reconstruct,locate,ply_bytes
from .tracker import Tracker
from .models import Models
from .store import Store


def png(rgb):
    out = io.BytesIO(); Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(out,format='PNG')
    return out.getvalue()


class Engine:
    def __init__(self,data_dir):
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True,exist_ok=True)
        self.store = Store(self.data_dir/'chimera.db')
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.models = Models()
        self.source = 'demo'; self.source_label = 'Synthetic RGB-D room'
        self.paused = False; self.capture = None; self.frame_index = 0
        self.tracker = Tracker(); self.session_id = self.store.session('demo')
        self.start_time = time.monotonic(); self.demo_time = 0.
        self.settings = {'stride':6,'confidence':.35,'fps':8}
        self.points = np.empty((0,6)); self.faces = []; self.tracks = []
        self.frames = {}; self.coverage = 0.; self.error = ''; self.fps = 0.; self.latency = 0.
        self.units = 'synthetic metres'; self.depth_source = 'Synthetic ground truth'
        self.last_saved = -10.; self.last_tick = time.monotonic(); self.fixed = None
        self.store.event(self.session_id,'info','Synthetic demonstration started — not live camera data')
        self.worker = threading.Thread(target=self.run,name='chimera-processor',daemon=True)
        self.worker.start()

    def close(self):
        self.stop_event.set(); self.worker.join(timeout=15)
        if self.worker.is_alive():
            return  # Do not close a database that a blocked camera worker may still use.
        with self.lock:
            if self.capture is not None: self.capture.release()
            self.store.close()

    def reset_state(self):
        self.tracker = Tracker(); self.tracks = []; self.points = np.empty((0,6)); self.faces = []
        self.frames = {}; self.coverage = 0.; self.error = ''; self.frame_index = 0; self.fps=0.
        self.demo_time = 0.; self.last_saved = -10.; self.start_time = time.monotonic()
        self.session_id = self.store.session(self.source)
        self.store.event(self.session_id,'source',f'Source opened: {self.source_label}')

    def set_source(self,kind,value=None):
        if kind not in ('demo','camera','video','rgbd'):
            raise ValueError('Unsupported source.')
        capture = None; fixed = None
        if kind in ('camera','video'):
            try: import cv2
            except ImportError as exc: raise ValueError('Install requirements-camera.txt to enable webcam and video.') from exc
            if kind == 'camera':
                index = int(value or 0)
                if not 0 <= index <= 12: raise ValueError('Camera index must be 0–12.')
                capture = cv2.VideoCapture(index)
            else:
                capture = cv2.VideoCapture(str(value))
            if not capture.isOpened():
                capture.release(); raise ValueError('Cannot open source. Check camera permission, index, or video codec.')
        if kind == 'rgbd':
            with np.load(value,allow_pickle=False) as data:
                rgb,depth = data['rgb'],data['depth']
                intr = data['intrinsics'].reshape(-1)
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]:
                    raise ValueError('NPZ requires rgb uint8 H×W×3 and aligned depth H×W.')
                if rgb.shape[0]*rgb.shape[1] > 3_000_000 or min(rgb.shape[:2]) < 16:
                    raise ValueError('RGB-D dimensions must be at least 16×16 and at most 3 megapixels.')
                if len(intr) != 4: raise ValueError('intrinsics must be [fx, fy, cx, cy].')
                if not np.issubdtype(depth.dtype,np.number): raise ValueError('Depth must be numeric metres.')
                k = dict(zip(('fx','fy','cx','cy'),map(float,intr)))
                reconstruct(rgb,depth,k,step=32)  # validate calibration before replacing source
                fixed = (rgb.copy(),depth.astype(np.float32),k)
        with self.lock:
            if self.capture is not None: self.capture.release()
            self.capture = capture; self.fixed = fixed; self.source = kind
            self.source_label = {'demo':'Synthetic RGB-D room','camera':f'Webcam {value or 0}','video':'Uploaded video','rgbd':'Calibrated RGB-D snapshot'}[kind]
            self.paused = False; self.reset_state()

    def configure(self,payload):
        with self.lock:
            stride = int(payload.get('stride',self.settings['stride']))
            confidence = float(payload.get('confidence',self.settings['confidence']))
            fps = int(payload.get('fps',self.settings['fps']))
            if not 3 <= stride <= 24 or not .05 <= confidence <= .95 or not 1 <= fps <= 20:
                raise ValueError('Stride 3–24, confidence 0.05–0.95, FPS 1–20 required.')
            if 'detector_path' in payload or 'depth_path' in payload:
                self.models.configure(payload.get('detector_path',''),payload.get('depth_path',''))
                self.tracker = Tracker(); self.tracks = []
            self.settings = {'stride':stride,'confidence':confidence,'fps':fps}
            self.store.event(self.session_id,'settings','Processing settings updated')

    def run(self):
        while not self.stop_event.is_set():
            tick = time.monotonic()
            try:
                with self.lock:
                    if not self.paused:
                        self.process()
            except Exception as exc:
                with self.lock:
                    self.error = f'{type(exc).__name__}: {exc}'
                    self.paused = True
                    self.store.event(self.session_id,'error',self.error)
            self.stop_event.wait(max(.005,1/self.settings['fps']-(time.monotonic()-tick)))

    def process(self):
        start = time.monotonic()
        now = start-self.start_time
        if self.source == 'demo':
            rgb,depth,detections,k = demo.scene(self.demo_time)
            self.demo_time += 1/self.settings['fps']
            self.units = 'synthetic metres'; self.depth_source = 'Synthetic ground truth'
        elif self.source == 'rgbd':
            rgb,depth,k = self.fixed
            detections = self.models.detect(rgb,self.settings['confidence'])
            self.units = 'metres'; self.depth_source = 'Provided calibrated depth'
        else:
            import cv2
            ok,frame = self.capture.read()
            if not ok:
                self.paused = True
                message = 'End of video. Reset to replay.' if self.source == 'video' else 'Camera read failed. Reopen the source.'
                self.store.event(self.session_id,'source',message)
                if self.source == 'camera': self.error=message
                return
            h,w = frame.shape[:2]
            scale = min(1.,640/w)
            if scale < 1: frame = cv2.resize(frame,(round(w*scale),round(h*scale)))
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            k = {'fx':float(w),'fy':float(w),'cx':w/2,'cy':h/2}
            detections = self.models.detect(rgb,self.settings['confidence'])
            depth = self.models.depth(rgb)
            self.units = 'relative units' if depth is not None else '2D pixels only'
            self.depth_source = 'Monocular estimate; assumed intrinsics' if depth is not None else 'No depth input'
        for d in detections:
            d['position'] = locate(d['box'],depth,k)
        self.tracks,events = self.tracker.update(detections,now)
        for kind,message in events: self.store.event(self.session_id,kind,message)
        if depth is not None:
            stride = max(self.settings['stride'],int(np.ceil(np.sqrt(depth.size/14000))))
            self.points,self.faces,self.coverage = reconstruct(rgb,depth,k,stride)
            valid = np.isfinite(depth)&(depth>.05)&(depth<20)
            z = np.where(valid,depth,0)
            dnorm = np.clip(1-z/max(.1,float(z.max())),0,1)
            dep = np.stack((dnorm*40,dnorm*180,dnorm*255),axis=-1).astype(np.uint8)
            dep[~valid] = 0
        else:
            self.points = np.empty((0,6)); self.faces = []; self.coverage=0.
            dep = np.zeros_like(rgb)
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)
        for o in self.tracks:
            color = tuple(demo.COLORS.get(o['label'],[34,169,255]))
            draw.rectangle(o['box'],outline=color,width=2)
            x,y = o['box'][:2]
            draw.text((x+2,max(0,y-12)),f"{o['label']} #{o['id']}",fill=color)
        self.frames = {'rgb':png(rgb),'depth':png(dep),'tracks':png(np.asarray(annotated))}
        self.frame_index += 1
        if now-self.last_saved >= 1:
            self.store.sample(self.session_id,now,self.tracks); self.last_saved=now
        self.latency = (time.monotonic()-start)*1000
        self.fps = 1/max(.001,start-self.last_tick); self.last_tick=start
        self.error = ''
        if self.source == 'rgbd': self.paused=True

    def state(self):
        with self.lock:
            return {'source':self.source,'source_label':self.source_label,'session':self.session_id,
                'paused':self.paused,'error':self.error,'frame':self.frame_index,'fps':round(self.fps,1),
                'latency_ms':round(self.latency,1),'points':len(self.points),'faces':len(self.faces),
                'coverage':round(self.coverage*100,1),'units':self.units,'depth_source':self.depth_source,
                'models':self.models.names,'tracks':self.tracks,'events':self.store.events(self.session_id),
                'settings':self.settings,'elapsed':round(time.monotonic()-self.start_time,1)}

    def cloud(self):
        with self.lock:
            return {'frame':self.frame_index,'points':np.round(self.points,4).tolist(),'faces':self.faces,'units':self.units}

    def export_ply(self):
        with self.lock:
            if not len(self.points): raise ValueError('No depth geometry to export.')
            return ply_bytes(self.points,self.faces)

    def control(self,action):
        with self.lock:
            if action == 'pause': self.paused=True
            elif action == 'resume': self.paused=False
            elif action == 'reset':
                if self.source == 'video':
                    import cv2
                    if not self.capture.set(cv2.CAP_PROP_POS_FRAMES,0): raise ValueError('Video does not support rewind.')
                self.reset_state(); self.paused=False
            else: raise ValueError('Unknown control action.')
