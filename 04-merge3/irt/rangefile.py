import requests, io
class RangeFile(io.RawIOBase):
    def __init__(self, url, sess=None):
        self.s = sess or requests.Session()
        r = self.s.get(url, headers={"Range":"bytes=0-0"}, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url; self.size = int(r.headers["Content-Range"].split("/")[1]); self.pos = 0; self.nreq=1
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else (self.pos+off if whence==1 else self.size+off); return self.pos
    def read(self, n=-1):
        if n is None or n<0: n = self.size-self.pos
        if n==0 or self.pos>=self.size: return b""
        end=min(self.size, self.pos+n)-1
        for a in range(4):
            try:
                r=self.s.get(self.url, headers={"Range":f"bytes={self.pos}-{end}"}, timeout=120); r.raise_for_status(); break
            except Exception:
                if a==3: raise
        self.nreq+=1; b=r.content; self.pos+=len(b); return b
    def readinto(self, b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)
