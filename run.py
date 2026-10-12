#!/usr/bin/env python3
import argparse
import threading
import webbrowser
from pathlib import Path
from chimera.engine import Engine
from chimera.server import Server


def main():
    parser=argparse.ArgumentParser(description='CHIMERA local 3D world intelligence workbench')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    args=parser.parse_args()
    engine=Engine(args.data_dir)
    try:
        server=Server(('127.0.0.1',args.port),engine)
    except OSError as exc:
        engine.close(); parser.exit(1,f'Cannot start server: {exc}. Try --port 8766\n')
    url=f'http://127.0.0.1:{args.port}'
    print(f'CHIMERA running at {url}\nPress Ctrl+C to stop. Demo data is synthetic.',flush=True)
    if not args.no_browser:
        opener=threading.Timer(.6,lambda:webbrowser.open(url)); opener.daemon=True; opener.start()
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: server.server_close(); engine.close()


if __name__ == '__main__': main()
