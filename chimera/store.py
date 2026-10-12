import sqlite3
import json
import threading
from datetime import datetime, timezone


class Store:
    def __init__(self,path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path,check_same_thread=False)
        self.db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, started TEXT, source TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session INTEGER, time TEXT, kind TEXT, message TEXT);
        CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, session INTEGER, time REAL, track INTEGER, label TEXT, x REAL, y REAL, z REAL, confidence REAL);''')
        self.db.commit()

    def session(self,source):
        with self.lock:
            cur = self.db.execute('INSERT INTO sessions(started,source) VALUES(?,?)',(datetime.now(timezone.utc).isoformat(),source))
            self.db.commit(); return cur.lastrowid

    def event(self,session,kind,message):
        with self.lock:
            self.db.execute('INSERT INTO events(session,time,kind,message) VALUES(?,?,?,?)',(session,datetime.now(timezone.utc).isoformat(),kind,message))
            self.db.commit()

    def sample(self,session,t,tracks):
        with self.lock:
            self.db.executemany('INSERT INTO samples(session,time,track,label,x,y,z,confidence) VALUES(?,?,?,?,?,?,?,?)',
                [(session,t,o['id'],o['label'],*(o['position'] or [None]*3),o['confidence']) for o in tracks])
            self.db.commit()

    def events(self,session):
        with self.lock:
            return [dict(zip(('id','time','kind','message'),r)) for r in self.db.execute('SELECT id,time,kind,message FROM events WHERE session=? ORDER BY id DESC LIMIT 30',(session,))]

    def sessions(self):
        with self.lock:
            return [dict(zip(('id','started','source'),r)) for r in self.db.execute('SELECT id,started,source FROM sessions ORDER BY id DESC LIMIT 100')]

    def samples(self,session):
        with self.lock:
            return self.db.execute('SELECT time,track,label,x,y,z,confidence FROM samples WHERE session=? ORDER BY id',(session,)).fetchall()

    def close(self):
        self.db.close()
