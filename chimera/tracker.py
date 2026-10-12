"""Class-aware IoU association, bounded history and lost-track expiry."""
from collections import deque
import math


def iou(a,b):
    x,y = max(a[0],b[0]),max(a[1],b[1])
    xx,yy = min(a[2],b[2]),min(a[3],b[3])
    overlap = max(0,xx-x)*max(0,yy-y)
    return overlap/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


class Tracker:
    def __init__(self, max_age=2.5):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, detections, now):
        self.tracks = {k:v for k,v in self.tracks.items() if now-v['seen'] <= self.max_age}
        pairs = []
        for k,t in self.tracks.items():
            for j,d in enumerate(detections):
                if d['label'] == t['label']:
                    score = iou(d['box'],t['box'])
                    if score > .12:
                        pairs.append((score,k,j))
        used_t,used_d = set(),set()
        assigned = {}
        for score,k,j in sorted(pairs,reverse=True):
            if k not in used_t and j not in used_d:
                assigned[j] = k
                used_t.add(k); used_d.add(j)
        visible,events = [],[]
        for j,d in enumerate(detections):
            k = assigned.get(j)
            if k is None:
                k = self.next_id; self.next_id += 1
                self.tracks[k] = {'id':k,'history':deque(maxlen=160),'last_event':now}
                events.append(('appeared',f"{d['label'].title()} #{k} observed"))
            t = self.tracks[k]
            old = t.get('position')
            pos = d.get('position')
            if old and pos and math.dist(old,pos) > .12 and now-t['last_event'] > 3:
                events.append(('movement',f"{d['label'].title()} #{k} moved"))
                t['last_event'] = now
            t.update(d); t['seen'] = now
            if pos:
                t['history'].append([round(now,3),*map(float,pos)])
            visible.append({**t,'history':list(t['history'])})
        return visible,events
