import unittest
import tempfile
import time
import threading
import json
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from chimera.geometry import reconstruct,locate,ply_bytes
from chimera.tracker import Tracker
from chimera.engine import Engine
from chimera.server import Server
from chimera import demo


class GeometryTests(unittest.TestCase):
    def test_pinhole_projection_and_mesh(self):
        rgb=np.full((6,6,3),120,dtype=np.uint8); depth=np.full((6,6),2.)
        points,faces,coverage=reconstruct(rgb,depth,{'fx':2,'fy':2,'cx':0,'cy':0},step=1)
        np.testing.assert_allclose(points[7,:3],[1,-1,2])
        self.assertEqual(coverage,1); self.assertEqual(len(faces),50)
        self.assertEqual(locate([0,0,2,2],depth,{'fx':2,'fy':2,'cx':0,'cy':0}),[1,-1,2])
        exported=ply_bytes(points,faces).decode()
        self.assertIn('element vertex 36',exported); self.assertIn('element face 50',exported)

    def test_invalid_depth_and_discontinuity(self):
        rgb=np.zeros((2,2,3),dtype=np.uint8)
        p,f,c=reconstruct(rgb,np.array([[1.,1.],[8.,8.]]),{'fx':2,'fy':2,'cx':0,'cy':0},1)
        self.assertEqual(f,[])
        p,f,c=reconstruct(rgb,np.array([[np.nan,0],[np.inf,2]]),{'fx':2,'fy':2,'cx':0,'cy':0},1)
        self.assertEqual(len(p),1); self.assertEqual(c,.25)
        with self.assertRaises(ValueError): reconstruct(rgb,np.ones((2,2)),{'fx':0,'fy':2,'cx':0,'cy':0})


class TrackerTests(unittest.TestCase):
    def test_stable_ids_class_and_expiry(self):
        tr=Tracker(max_age=2)
        def d(label='person',shift=0): return {'label':label,'box':[shift,0,20+shift,20],'position':[1,2,3],'confidence':.9}
        first,_=tr.update([d()],0); key=first[0]['id']
        second,_=tr.update([d(shift=1)],.5); self.assertEqual(second[0]['id'],key)
        other,_=tr.update([d('chair')],1); self.assertNotEqual(other[0]['id'],key)
        new,_=tr.update([d()],5); self.assertNotEqual(new[0]['id'],key)

    def test_one_to_one_and_history_bound(self):
        tr=Tracker()
        d={'label':'person','box':[0,0,50,50],'position':[1,2,3],'confidence':.8}
        for i in range(220): visible,_=tr.update([d.copy(),d.copy()],i*.1)
        self.assertEqual(len({t['id'] for t in visible}),2)
        self.assertEqual(len(visible[0]['history']),160)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.engine=Engine(self.tmp.name)
        self.server=Server(('127.0.0.1',0),self.engine)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        for _ in range(100):
            if self.engine.state()['frame']>0: break
            time.sleep(.03)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.engine.close();self.tmp.cleanup()
    def get(self,path): return urllib.request.urlopen(self.base+path,timeout=10)
    def post(self,path,data,headers=None):
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode(),headers=headers or {'Content-Type':'application/json'})
        return urllib.request.urlopen(req,timeout=10)
    def test_demo_pause_reset_and_exports(self):
        state=json.load(self.get('/api/state'));self.assertEqual(state['error'],'');self.assertGreater(state['points'],500)
        self.assertGreater(len(state['tracks']),0)
        self.post('/api/control',{'action':'pause'})
        frame=self.engine.state()['frame'];time.sleep(.2);self.assertEqual(self.engine.state()['frame'],frame)
        self.assertIn(b'ply\n',self.get('/api/export/scene.ply').read())
        self.assertIn(b'track_id',self.get('/api/export/tracks.csv').read())
        self.assertEqual(self.get('/api/frame/rgb').headers.get_content_type(),'image/png')
        self.post('/api/control',{'action':'reset'});self.assertGreater(self.engine.state()['session'],state['session'])
    def test_bad_source_settings_and_origin(self):
        for route,payload in [('/api/source',{'kind':'file','path':'/etc/passwd'}),('/api/settings',{'stride':0}),('/api/control',{'action':'bad'})]:
            with self.assertRaises(urllib.error.HTTPError) as error:self.post(route,payload)
            self.assertEqual(error.exception.code,400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/control',{'action':'pause'},{'Origin':'https://evil.invalid'})
        self.assertEqual(error.exception.code,403)
        with self.assertRaises(urllib.error.HTTPError):self.get('/../run.py')
    def test_video_decode_eof_and_rewind(self):
        try: import cv2
        except ImportError: self.skipTest('OpenCV not installed')
        path=Path(self.tmp.name)/'video.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
        if not writer.isOpened(): self.skipTest('MJPG encoder unavailable')
        for i in range(4): writer.write(np.full((48,64,3),i*50,dtype=np.uint8))
        writer.release()
        self.engine.set_source('video',path)
        for _ in range(100):
            if self.engine.state()['paused']:break
            time.sleep(.03)
        state=self.engine.state()
        self.assertEqual(state['error'],'');self.assertTrue(state['paused'])
        self.assertEqual(state['frame'],4);self.assertEqual(state['points'],0)
        self.assertEqual(state['tracks'],[]);self.assertEqual(state['units'],'2D pixels only')
        self.engine.control('reset')
        for _ in range(100):
            if self.engine.state()['paused']:break
            time.sleep(.03)
        self.assertEqual(self.engine.state()['frame'],4)

    def test_rgbd_import_roundtrip_and_invalid_shape(self):
        rgb,depth,_,k=demo.scene(0);path=Path(self.tmp.name)/'frame.npz'
        np.savez_compressed(path,rgb=rgb,depth=depth,intrinsics=list(k.values()))
        req=urllib.request.Request(self.base+'/api/upload',data=path.read_bytes(),headers={'X-Filename':'room.npz','Content-Type':'application/octet-stream'})
        result=json.load(urllib.request.urlopen(req));self.assertEqual(result['source'],'rgbd')
        for _ in range(100):
            if self.engine.state()['frame']>0:break
            time.sleep(.03)
        state=self.engine.state();self.assertEqual(state['units'],'metres');self.assertTrue(state['paused']);self.assertGreater(state['points'],500)
        np.savez(path,rgb=rgb,depth=np.ones((3,3)),intrinsics=list(k.values()))
        with self.assertRaises(ValueError):self.engine.set_source('rgbd',path)
        self.assertEqual(self.engine.source,'rgbd')


if __name__=='__main__':unittest.main()
