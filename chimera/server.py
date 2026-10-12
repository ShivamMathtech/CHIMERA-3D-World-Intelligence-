"""Loopback-only HTTP server; same-origin writes and bounded uploads."""
import csv
import io
import json
import mimetypes
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .engine import Engine

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 256*1024*1024


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self,address,engine):
        self.engine=engine
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','304'): super().log_message(format,*args)

    def reply(self,data,status=200,mime='application/json',filename=None):
        if isinstance(data,(dict,list)): data=json.dumps(data,allow_nan=False).encode()
        if isinstance(data,str): data=data.encode()
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename: self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError): pass

    def check_origin(self):
        host = self.headers.get('Host','')
        allowed = {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if host not in allowed: raise PermissionError('Invalid local Host header.')
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            raise PermissionError('Cross-origin requests are blocked.')

    def do_GET(self):
        try:
            self.check_origin()
            url=urlparse(self.path); path=url.path; engine=self.server.engine
            if path == '/api/state': return self.reply(engine.state())
            if path == '/api/cloud': return self.reply(engine.cloud())
            if path == '/api/sessions': return self.reply(engine.store.sessions())
            if path.startswith('/api/frame/'):
                with engine.lock: data=engine.frames.get(path.rsplit('/',1)[-1])
                return self.reply(data, mime='image/png') if data else self.reply({'error':'No frame yet'},404)
            if path == '/api/export/scene.ply': return self.reply(engine.export_ply(),mime='application/octet-stream',filename='chimera-scene.ply')
            if path == '/api/export/tracks.csv':
                session=int(parse_qs(url.query).get('session',[engine.session_id])[0])
                out=io.StringIO(); writer=csv.writer(out)
                writer.writerow(['time_s','track_id','class','x','y','z','confidence'])
                writer.writerows(engine.store.samples(session))
                return self.reply(out.getvalue(),mime='text/csv',filename=f'chimera-session-{session}-tracks.csv')
            if path == '/api/export/session.json':
                return self.reply(engine.state(),mime='application/json',filename='chimera-session.json')
            target=(ROOT/'web'/('index.html' if path == '/' else path.lstrip('/'))).resolve()
            if not target.is_relative_to((ROOT/'web').resolve()) or not target.is_file(): return self.reply({'error':'Not found'},404)
            return self.reply(target.read_bytes(),mime=mimetypes.guess_type(target)[0] or 'application/octet-stream')
        except PermissionError as exc: self.reply({'error':str(exc)},403)
        except (ValueError,KeyError) as exc: self.reply({'error':str(exc)},400)
        except Exception as exc: self.reply({'error':f'{type(exc).__name__}: {exc}'},500)

    def do_POST(self):
        try:
            self.check_origin()
            path=urlparse(self.path).path; engine=self.server.engine
            size=int(self.headers.get('Content-Length','0'))
            limit=MAX_UPLOAD if path == '/api/upload' else 16384
            if not 0 < size <= limit: return self.reply({'error':f'Body required; maximum {limit} bytes'},413)
            self.connection.settimeout(60)
            if path == '/api/upload':
                suffix=Path(self.headers.get('X-Filename','')).suffix.lower()
                if suffix not in ('.mp4','.avi','.mov','.mkv','.webm','.npz'): raise ValueError('Upload video or RGB-D .npz.')
                dest=engine.data_dir/'uploads'/f'{uuid.uuid4().hex}{suffix}'; dest.parent.mkdir(exist_ok=True)
                try:
                    with dest.open('wb') as f:
                        remaining=size
                        while remaining:
                            chunk=self.rfile.read(min(1024*1024,remaining))
                            if not chunk: raise ValueError('Incomplete upload.')
                            f.write(chunk); remaining-=len(chunk)
                    if suffix == '.npz':
                        with zipfile.ZipFile(dest) as z:
                            if sum(x.file_size for x in z.infolist()) > 64*1024*1024: raise ValueError('Expanded NPZ exceeds 64 MB.')
                    engine.set_source('rgbd' if suffix=='.npz' else 'video',dest)
                except Exception:
                    dest.unlink(missing_ok=True); raise
                return self.reply({'ok':True,'source':engine.source})
            payload=json.loads(self.rfile.read(size))
            if not isinstance(payload,dict): raise ValueError('JSON object required.')
            if path == '/api/control': engine.control(payload['action'])
            elif path == '/api/source':
                if payload.get('kind') not in ('demo','camera'): raise ValueError('Use upload for video or RGB-D files.')
                engine.set_source(payload['kind'],payload.get('index',0))
            elif path == '/api/settings': engine.configure(payload)
            else: return self.reply({'error':'Not found'},404)
            self.reply({'ok':True})
        except PermissionError as exc: self.reply({'error':str(exc)},403)
        except (ValueError,KeyError,ImportError,OSError,zipfile.BadZipFile) as exc: self.reply({'error':str(exc)},400)
        except Exception as exc: self.reply({'error':f'{type(exc).__name__}: {exc}'},500)
