# -*- coding: utf-8 -*-
# -------- Unwanted Creatures RTSP Detector (10 fps, high-quality capture) + SORT Tracking --------

import os
import cv2
import time
import uuid
import math
import socket
import threading
import requests
from datetime import datetime
from collections import defaultdict, deque
import signal
from zoneinfo import ZoneInfo
import fcntl
import struct
import queue  # สำหรับ timeout ตอนเปิด RTSP

from ultralytics import YOLO
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from google.auth.transport.requests import Request

import numpy as np
from simple_sort import Sort  # ใช้ SORT แบบ pure-Python

# ===================== CONFIG =====================
TARGET_FPS = 10
RTSP_URL = "rtsp://Project:Project1234@10.44.63.127/stream1"
USE_TCP = True 
YOLO_WEIGHTS = "best.pt"
YOLO_CONF = 0.5
INTEREST_LABELS = {"centipede", "lizard", "mouse", "snake"}

THAI_TRANSLATIONS = {
    "centipede": "ตะขาบ",
    "lizard": "ตัวเงินตัวทอง",
    "mouse": "หนู",
    "snake": "งู"
}

DETECT_EVERY_N_FRAMES = 2
USE_PYAV = True # ใช้ไลบรารี PyAV ในการอ่าน RTSP (ให้ timestamp ที่แม่นยำกว่า)
TIMESTAMP_OFFSET_SECONDS = 0.0
LOCAL_TZ = ZoneInfo("Asia/Bangkok")
SHOW_WINDOW = True
WINDOWS_SD_CANDIDATES = ["/boot/firmware/detections"]
MAX_SD_MB = 300
MAX_SD_FILES = 2000


TRACK_MAX_AGE = 15 # จำนวนเฟรมสูงสุดที่ tracker จะยังจำ object ที่หายไปได้
TRACK_MIN_HITS = 2 # จำนวนครั้งขั้นต่ำที่ต้องเจอ object ก่อนจะเริ่ม track
TRACK_IOU_THRESH = 0.2 # ค่า IoU threshold สำหรับการจับคู่ object เดิม

# เวลา timeout (วินาที)
OPEN_TIMEOUT = 8.0         # timeout ตอน "เปิด RTSP"
FIRST_FRAME_TIMEOUT = 8.0  # timeout ตอนรอ "เฟรมแรก"

cloudinary.config(
    cloud_name="detcb8wc4",
    api_key="976789947296377",
    api_secret="6vmT6GZvtoBfrShhR7oXfyGSl90"
)
# ===================================================

# ===== GPIO (จำลองได้ถ้าไม่มีฮาร์ดแวร์) =====
try:
    from gpiozero import PWMLED
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

class SafeRGB:
    def __init__(self, pin_r=17, pin_g=27, pin_b=22, invert=False):
        self.INVERT = invert
        self.enabled = _HAS_GPIO
        self.r = self.g = self.b = None
        if not self.enabled:
            print("[WARN] gpiozero not available. Simulated LED mode."); return
        try:
            self.r = PWMLED(pin_r, active_high=not invert, frequency=1000)
            self.g = PWMLED(pin_g, active_high=not invert, frequency=1000)
            self.b = PWMLED(pin_b, active_high=not invert, frequency=1000)
        except Exception as e:
            print(f"[WARN] PWMLED init failed -> Simulated LED mode. Detail: {e}")
            self.enabled = False

    def _apply(self, rr, gg, bb):
        if not self.enabled:
            print(f"[LED] R={rr} G={gg} B={bb}"); return
        self.r.value, self.g.value, self.b.value = rr, gg, bb

    def off(self): self._apply(0,0,0)
    def red(self): self._apply(1,0,0)
    def green(self): self._apply(0,1,0)
    def blue(self): self._apply(0,0,1)

    def cleanup(self):
        try: self.off()
        except Exception: pass

rgb = SafeRGB(pin_r=17, pin_g=27, pin_b=22, invert=False)

def _rgb_self_test():
    try:
        print("[LED] self-test: R, G, B, White, Off")
        rgb.red(); time.sleep(0.2)
        rgb.green(); time.sleep(0.2)
        rgb.blue(); time.sleep(0.2)
        rgb._apply(1,1,1); time.sleep(0.2)
        rgb.red()
    except Exception as e:
        print("[LED] self-test failed:", e)
_rgb_self_test()

# --- เขียวกระพริบระหว่างต่อสตรีม ---
def _blink_green_until(stop_event: threading.Event, interval: float = 0.3):
    """กระพริบไฟสีเขียวจนกว่า stop_event จะถูก set"""
    while not stop_event.is_set():
        try:
            rgb.green()
            time.sleep(interval)
            rgb.off()
            time.sleep(interval)
        except Exception:
            time.sleep(interval)

# --- เช็คอินเทอร์เน็ต (แดงค้างเมื่อไม่มีเน็ต) ---
def has_internet(timeout: float = 1.5) -> bool:
    """คืน True ถ้ามีเส้นทางออกอินเทอร์เน็ต (ลอง connect UDP ไป 8.8.8.8:53)"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.connect(("8.8.8.8", 53))
        s.close()
        rgb.green()
        return True
    except Exception:
        rgb.red()
        return False

# ===== Save helpers =====
_cached_windows_sd_dir = None

# ฟังก์ชันสำหรับลบไฟล์เก่าทิ้งเมื่อพื้นเต็มที่หรือจำนวนไฟล์เกินกำหนด
def _prune_if_needed(dest_dir):
    try:
        files=[]
        for name in os.listdir(dest_dir):
            fp=os.path.join(dest_dir,name)
            if os.path.isfile(fp): files.append((fp, os.path.getmtime(fp), os.path.getsize(fp)))
        files.sort(key=lambda x:x[1])

        # ลบไฟล์ที่เก่าที่สุดออก ถ้าจำนวนไฟล์เกินกำหนด
        while len(files)>MAX_SD_FILES:
            fp,*_=files.pop(0)
            try: os.remove(fp); print(f"[PRUNE] by count: {fp}")
            except Exception as e: print(f"[PRUNE] remove failed: {fp} -> {e}")
        size_mb=sum(sz for _,__,sz in files)/(1024*1024)

        # ลบไฟล์ที่เก่าที่สุดออก ถ้าขนาดพื้นที่เกินกำหนด
        while size_mb>MAX_SD_MB and files:
            fp,_,sz=files.pop(0)
            try: os.remove(fp); size_mb-=sz/(1024*1024); print(f"[PRUNE] by size: {fp}")
            except Exception as e: print(f"[PRUNE] remove failed: {fp} -> {e}")
    except Exception as e:
        print(f"[PRUNE] error: {e}")
        
# ฟังก์ชันหาตำแหน่งที่จะบันทึกไฟล์ภาพ
# จะพยายามใช้ SD Card ที่เสียบอยู่ (สำหรับ Raspberry Pi) ก่อน
# หากไม่เจอ จะใช้โฟลเดอร์ในเครื่องแทน
def get_windows_sd_dir():
    global _cached_windows_sd_dir
    if _cached_windows_sd_dir: return _cached_windows_sd_dir
    for path in WINDOWS_SD_CANDIDATES:
        base=os.path.dirname(path)
        if os.path.exists(base):
            try:
                os.makedirs(path, exist_ok=True)
                _cached_windows_sd_dir=path
                print(f"[SAVE] Using BOOT folder: {path}")
                return path
            except Exception as e:
                print(f"[SAVE] Cannot create {path}: {e}")
    fallback=os.path.join(os.getcwd(),"detections_local")
    os.makedirs(fallback, exist_ok=True)
    print(f"[SAVE] BOOT partition not writable. Fallback: {fallback}")
    _cached_windows_sd_dir=fallback
    return fallback

# ===== Firebase init =====
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("ServiceAccountKey.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()
db = init_firebase()

def get_ip_address(ifname='eth0'):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', ifname[:15].encode('utf-8'))
        )[20:24])
        print(f"[IP] Found IP for {ifname}: {addr}")
        rgb.red()
        return addr
    except Exception:
        print(f"[IP] Could not get IP for {ifname}. Falling back...")
        try:
            s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(0.2)
            s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]; s.close()
            print(f"[IP] Fallback success (outbound): {ip}")
            return ip
        except Exception:
            try:
                ip = socket.gethostbyname(socket.gethostname())
                print(f"[IP] Fallback success (hostname): {ip}")
                return ip
            except Exception:
                rgb.red()
                print("[IP] All methods failed.")
                return "unknown"

ip_address = get_ip_address()
existing = db.collection("Raspberry_pi").where("ip","==",ip_address).get()
if existing:
    pi_doc_ref = db.collection("Raspberry_pi").document(existing[0].id)
else:
    new_doc_ref = db.collection("Raspberry_pi").document()
    new_doc_ref.set({"ip":ip_address,"id":new_doc_ref.id,"createdAt":firestore.SERVER_TIMESTAMP})
    pi_doc_ref = new_doc_ref

# ===== FCM v1 (ฟังก์ชันสำหรับส่ง Push Notification ไปยังมือถือ) =====
def send_fcm_v1_notification(title, body, topic):
    if not topic:
        print("[FCM] No topic provided, skipping notification.")
        return
    try:
        sa = service_account.Credentials.from_service_account_file(
            "ServiceAccountKey.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        sa.refresh(Request()); access_token = sa.token
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json; UTF-8"}
        project_id = sa.project_id
        url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
        message_payload = {"message":{"topic":topic,"notification":{"title":title,"body":body},
                                     "android":{"priority":"high","notification":{"channel_id":"high_importance_channel"}}}}
        resp = requests.post(url, headers=headers, json=message_payload, timeout=8)
        if resp.status_code>=300: print(f"[FCM] error [{resp.status_code}] sending to topic '{topic}': {resp.text[:200]}")
        else: print(f"[FCM] sent to topic '{topic}'")
    except Exception as e:
        rgb.red()
        print("[FCM] send failed:", e)

# ===== Upload worker =====
def threaded_upload(image_path, detections_list, captured_ts):
    def upload():
        if not detections_list:
            return
        try:
            if not os.path.exists(image_path):
                print(f"[UPLOAD] file not found: {image_path}"); return
            print(f"[UPLOAD] start event upload: {image_path}")
            results = cloudinary.uploader.upload(image_path, folder="unwanted-creatures", timeout=30)
            image_url = results.get('secure_url')
            if not image_url:
                print("[UPLOAD] no secure_url returned"); return

            size_kb = round(os.path.getsize(image_path)/1024, 2)
            stamp = datetime.fromtimestamp(captured_ts, tz=LOCAL_TZ)
            doc_id = stamp.strftime("%Y-%m-%d_%H-%M-%S") + f"_{uuid.uuid4().hex[:4]}"

            detected_objects_data = [
                {"type": d['label'], "confidence": float(d['conf']), "track_id": int(d['track_id'])}
                for d in detections_list
            ]

            db.collection("Raspberry_pi").document(pi_doc_ref.id).collection("detections").document(doc_id).set({
                "detected_objects": detected_objects_data,
                "image_url": image_url,
                "image_size_kb": size_kb,
                "captured_at_epoch": float(captured_ts),
                "captured_at_str": stamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
                "createdAt": firestore.SERVER_TIMESTAMP,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
            print(f"[UPLOAD] success (event): {image_url}")

            device_doc = pi_doc_ref.get()
            if device_doc.exists:
                device_data = device_doc.to_dict()
                owner_id = device_data.get("ownerId")
                status = device_data.get("status")
                
                # จะส่ง notification ก็ต่อเมื่อมี ownerId และ status เป็น "online"
                if owner_id and status == "online":
                    counts = defaultdict(int)
                    for det in detections_list:
                        counts[det['label']] += 1
                    body_parts = []
                    for label, count in sorted(counts.items()):
                        thai_label = THAI_TRANSLATIONS.get(label, label.capitalize())
                        body_parts.append(f"{thai_label} {count} ตัว")
                    notification_body = "ตรวจพบ " + ", ".join(body_parts)
                    notification_title = "ตรวจพบสิ่งมีชีวิตไม่พึงประสงค์"
                    send_fcm_v1_notification(
                        title=notification_title,
                        body=notification_body,
                        topic=owner_id
                    )
                else:
                    print(f"[FCM] Skipping notification. Owner: {owner_id}, Status: {status}")
            else:
                print("[FCM] Device document not found. Notification not sent.")
        except Exception as e:
            rgb.red()
            print("[UPLOAD] failed:", e)

    threading.Thread(target=upload, daemon=True).start()

# ===== Overlay (วาดกรอบผลตรวจ + track id) =====
class DetOverlay:
    def __init__(self, ttl=2.0):
        self.lock=threading.Lock(); self.ttl=ttl; self.items=[]
    def update(self, dets):
        now=time.time()
        with self.lock: self.items=[{**d,"ts":now} for d in dets]
    def draw(self, frame):
        now=time.time()
        with self.lock: fresh=[d for d in self.items if (now-d["ts"])<=self.ttl]
        COLORS={"snake":(0,255,0),"mouse":(255,0,0),"centipede":(0,255,255),"lizard":(255,255,0)}
        for d in fresh:
            x1,y1,x2,y2=d["x1"],d["y1"],d["x2"],d["y2"]
            label,conf=d.get("label","object"),d.get("conf",0.0)
            tid=d.get("track_id",-1)
            color=COLORS.get(label,(0,255,0))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            txt=f"ID {tid} | {label} {conf:.2f}"
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            y_text=max(0,y1-8)
            cv2.rectangle(frame,(x1,y_text-th-6),(x1+tw+6,y_text+2),color,-1)
            cv2.putText(frame,txt,(x1+3,y_text),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

det_overlay = DetOverlay(ttl=2.0)

# ===== MoveGate (กลไกกรองการแจ้งเตือนตาม 'การเคลื่อนที่') =====
# วัตถุประสงค์: ป้องกันการส่งแจ้งเตือนซ้ำๆ สำหรับ object เดิมที่ขยับแค่นิดเดียว
# หลักการทำงาน: จะส่งแจ้งเตือนครั้งแรกที่เจอ object และจะส่งอีกครั้งก็ต่อเมื่อ object นั้น
# เคลื่อนที่ออกจากจุดที่ส่งแจ้งเตือนล่าสุดไปเป็นระยะทางที่กำหนด (move_thresh)
def _norm_dist(p1,p2,w,h):
    dx=(p1[0]-p2[0])/w; dy=(p1[1]-p2[1])/h; return math.hypot(dx,dy)

class MoveGate:
    def __init__(self, snap_thresh=0.03, move_thresh=0.12, track_timeout=3.0, hist_len=10):
        self.snap_thresh=snap_thresh # ระยะที่จะถือว่าเป็น object ตัวเดิม
        self.move_thresh=move_thresh # ระยะเคลื่อนที่ขั้นต่ำที่จะ trigger การส่งใหม่
        self.track_timeout=track_timeout # เวลาที่ object หายไปแล้วจะไม่จำ
        self.tracks=defaultdict(list)
        self.hist_len = hist_len
    def _prune(self, now):
        # ลบ track ที่เก่าเกินไป
        for label in list(self.tracks.keys()):
            self.tracks[label]=[t for t in self.tracks[label] if now-t["last_seen"]<=self.track_timeout]
            if not self.tracks[label]: del self.tracks[label]
    def update_and_should_send(self,label,cx,cy,fw,fh,now=None):
        if now is None: now=time.time()
        self._prune(now)
        best_i,best_d=None,1e9
        for i,t in enumerate(self.tracks[label]):
            d=_norm_dist((cx,cy),t["pos"],fw,fh)
            if d<best_d: best_d,best_i=d,i
        if best_i is not None and best_d<=self.snap_thresh:
            t=self.tracks[label][best_i]
            t["pos"]=(cx,cy); t["last_seen"]=now
        else:
            t={"pos":(cx,cy),"last_seen":now,"last_sent_pos":None,"hist":deque(maxlen=self.hist_len)}
            self.tracks[label].append(t)
        t["hist"].append((cx,cy))
        if t["last_sent_pos"] is None:
            t["last_sent_pos"]=t["pos"]; moved_norm=0.0; is_stationary=True
            return True, t, is_stationary, moved_norm
        moved_norm=_norm_dist(t["pos"],t["last_sent_pos"],fw,fh)
        if len(t["hist"]) >= 2:
            dsum=0.0; cnt=0
            for (x0,y0),(x1,y1) in zip(list(t["hist"])[:-1], list(t["hist"])[1:]):
                dsum += _norm_dist((x0,y0),(x1,y1),fw,fh); cnt += 1
            avg_move = dsum/max(1,cnt)
        else:
            avg_move = 0.0
        is_stationary = avg_move < 0.02
        if moved_norm>=self.move_thresh:
            t["last_sent_pos"]=t["pos"]
            return True, t, is_stationary, moved_norm
        return False, t, is_stationary, moved_norm

move_gate = MoveGate()

# ===== AlertGate (กลไกป้องกันการแจ้งเตือนซ้ำซ้อน 'ในพื้นที่และเวลาเดิม') =====
# แบ่งภาพเป็นช่อง Grid และมี Cooldown สำหรับแต่ละช่อง
# ป้องกันการแจ้งเตือนถี่ๆ เมื่อมีสัตว์ตัวเดิมวนเวียนอยู่ในบริเวณเดียวกัน
class AlertGate:
    def __init__(self, grid=20, cooldown_stationary=120.0, cooldown_moving=30.0):
        self.grid = grid
        self.cooldown_stationary = cooldown_stationary
        self.cooldown_moving = cooldown_moving
        self.last_alert = {}
    def _key(self, label, cx, cy, fw, fh):
        gx = max(0, min(self.grid-1, int(self.grid * (cx / max(1.0, fw)))))
        gy = max(0, min(self.grid-1, int(self.grid * (cy / max(1.0, fh)))))
        return (label, gx, gy)
    def allow(self, label, cx, cy, fw, fh, is_stationary, now=None):
        if now is None: now = time.time()
        key = self._key(label, cx, cy, fw, fh)
        last = self.last_alert.get(key, 0.0)
        cd = self.cooldown_stationary if is_stationary else self.cooldown_moving
        if now - last >= cd:
            self.last_alert[key] = now
            return True, cd
        return False, cd

alert_gate = AlertGate(grid=20, cooldown_stationary=120.0, cooldown_moving=30.0)

# ===== YOLO + SORT tracker =====
model = YOLO(YOLO_WEIGHTS)
tracker = Sort(max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS, iou_threshold=TRACK_IOU_THRESH)

# ===== median lag helper =====
class MedianLag:
    def __init__(self, maxlen=120):
        self.buf = deque(maxlen=maxlen)
        self._value = 0.0
    def update(self, lag_seconds: float):
        self.buf.append(float(lag_seconds))
        if len(self.buf) >= 5:
            arr = sorted(self.buf)
            n = len(arr); m = n//2
            self._value = arr[m] if (n % 2 == 1) else 0.5*(arr[m-1]+arr[m])
    @property
    def value(self) -> float:
        return self._value
        
# ===== RTSP Reader (Class หลักสำหรับอ่านสตรีมวิดีโอ) =====
# *** จุดสำคัญ: อ่านเฟรมใน Thread แยก, พยายามต่อใหม่อัตโนมัติ, ใช้ PyAV เพื่อความแม่นยำ ***
class RTSPReader:
    def __init__(self, url, use_tcp=True):
        self.url = url
        self.use_tcp = use_tcp
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_ts = None
        self._mode = "opencv"
        self._lag = MedianLag(maxlen=120)

        if USE_PYAV:
            try:
                import av
                self._av = av
                self._mode = "pyav"
                print("[RTSP] Using PyAV reader")
            except Exception as e:
                print(f"[RTSP] PyAV not available ({e}), falling back to OpenCV.")

        if self._mode == "pyav":
            self._open_pyav()
            self.t = threading.Thread(target=self._loop_pyav, daemon=True); self.t.start()
        else:
            self._open_cv()
            self.t = threading.Thread(target=self._loop_cv, daemon=True); self.t.start()
            
    # ส่วนของ PyAV จะมีการคำนวณ timestamp ที่ซับซ้อนกว่า แต่แม่นยำกว่า
    def _open_pyav(self):
        opts = {
            "rtsp_transport": "tcp" if self.use_tcp else "udp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "reorder_queue_size": "0",
            "rtbufsize": "0",
            "stimeout": "5000000",
            "rw_timeout": "5000000",
            "use_wallclock_as_timestamps": "0",
        }
        self.av_container = self._av.open(self.url, options=opts, timeout=5)
        vstreams = [s for s in self.av_container.streams if s.type == "video"]
        if not vstreams:
            raise RuntimeError("No video stream in RTSP")
        self.vstream = vstreams[0]
        self.vstream.thread_type = "AUTO"

    def _loop_pyav(self):
        stream_t0 = None
        wall_t0   = None
        while not self.stopped:
            try:
                for packet in self.av_container.demux(self.vstream):
                    if self.stopped: break
                    for frame in packet.decode():
                        if frame.time is not None:
                            stream_t = float(frame.time)
                        else:
                            pts = frame.pts; tb = frame.time_base
                            stream_t = float(pts * tb) if (pts is not None and tb is not None) else None

                        now_wall = time.time()
                        if stream_t is None:
                            ts_raw = now_wall
                        else:
                            if stream_t0 is None:
                                stream_t0 = stream_t; wall_t0 = now_wall
                            ts_raw = wall_t0 + (stream_t - stream_t0)

                        self._lag.update(now_wall - ts_raw)
                        ts_corr = ts_raw - self._lag.value + TIMESTAMP_OFFSET_SECONDS

                        img = frame.to_ndarray(format="bgr24")
                        with self.lock:
                            self.frame_ts = (img, ts_corr)
                    if self.stopped: break
            except Exception as e:
                # ถ้าหลุด จะพยายามต่อใหม่
                print(f"[RTSP][PyAV] loop error: {e}")
                time.sleep(0.5)
                try:
                    self._open_pyav()
                    stream_t0 = None; wall_t0 = None
                except Exception as e2:
                    rgb.red()
                    print(f"[RTSP][PyAV] reopen failed: {e2}")
                    time.sleep(1)
                    
    # ส่วนของ OpenCV จะใช้ง่ายกว่า แต่ timestamp จะเป็นเวลาที่อ่านเฟรมได้ ไม่ใช่เวลาจริงของเฟรม
    def _open_cv(self):
        transport = "tcp" if self.use_tcp else "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}"
            "|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
            "|max_delay;500000|stimeout;5000000|rw_timeout;5000000|rtbufsize;0"
            "|use_wallclock_as_timestamps;0"
        )
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _loop_cv(self):
        fail=0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5); self._open_cv(); continue
            ret,f=self.cap.read()
            if not ret:
                fail+=1
                if fail>=20:
                    try: self.cap.release()
                    except: pass
                    self.cap=None; fail=0
                time.sleep(0.02); continue
            fail=0
            ts = time.time() + TIMESTAMP_OFFSET_SECONDS
            with self.lock:
                self.frame_ts=(f, ts)

    def read(self):
        with self.lock:
            if self.frame_ts is None: return None
            f, ts = self.frame_ts
            return (f.copy(), ts)

    def release(self):
        self.stopped=True
        if getattr(self, "cap", None) is not None:
            try: self.cap.release()
            except: pass
        if getattr(self, "av_container", None) is not None:
            try: self.av_container.close()
            except: pass

# --- helper สำหรับเปิด RTSP แบบมี timeout ---
# ป้องกันไม่ให้โปรแกรมค้าง ถ้าหากกล้องปิดอยู่หรือไม่สามารถเชื่อมต่อได้ตอนเริ่ม
def _make_reader_with_timeout(url, use_tcp=True, timeout=OPEN_TIMEOUT):
    """พยายามสร้าง RTSPReader ภายในเวลาที่กำหนด -> (reader, err)"""
    q = queue.Queue(maxsize=1)
    def builder():
        try:
            r = RTSPReader(url, use_tcp=use_tcp)
            q.put(("ok", r))
        except Exception as e:
            q.put(("err", e))
    th = threading.Thread(target=builder, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None, "timeout while opening RTSP"
    try:
        status, payload = q.get_nowait()
    except queue.Empty:
        return None, "unknown error (no result)"
    if status == "ok":
        return payload, None
    else:
        return None, payload  # payload = Exception

# ===== Main loop/control =====
_STOP = threading.Event()

def run_detection_loop():
    # เขียวกระพริบระหว่าง "กำลังเปิด/ดึง RTSP"
    connect_stop = threading.Event()
    blinker = threading.Thread(target=_blink_green_until, args=(connect_stop, 0.3), daemon=True)
    blinker.start()

    # 1) เปิด RTSP แบบมี timeout ระดับ constructor
    reader, err = _make_reader_with_timeout(RTSP_URL, use_tcp=USE_TCP, timeout=OPEN_TIMEOUT)
    if reader is None:
        connect_stop.set(); blinker.join(timeout=1.0)
        rgb.red()
        _fatal_red_halt(f"[RTSP] open failed at constructor: {err}")
        return

    # 2) รอเฟรมแรกแบบมี timeout
    t0 = time.monotonic()
    while reader.read() is None and (time.monotonic() - t0) < FIRST_FRAME_TIMEOUT:
        time.sleep(0.05)

    if reader.read() is None:
        connect_stop.set(); blinker.join(timeout=1.0)
        reader.release()
        _fatal_red_halt("[RTSP] no first frame within timeout")
        return

    # ต่อ RTSP สำเร็จ → เขียวค้าง
    connect_stop.set(); blinker.join(timeout=1.0)
    print("[RTSP] reading... (q=quit)")
    rgb.green()

    frame_period = 1.0 / TARGET_FPS
    last_time = time.time()
    frame_idx = 0

    last_detections_xyxy = np.empty((0, 5), dtype=float)
    last_labels = []
    
    #คำนวณค่า Intersection over Union (IoU) ระหว่าง bounding box สองกล่อง (a และ b)
    #IoU เป็นตัวชี้วัดว่ากล่องสองใบซ้อนทับกันมากแค่ไหน
    #- ค่าเข้าใกล้ 1 หมายถึงซ้อนทับกันสนิท
    #- ค่าเป็น 0 หมายถึงไม่ซ้อนทับกันเลย
    def iou(a, b):
        # ดึงค่าพิกัด (มุมซ้ายบน และ มุมขวาล่าง) ของกล่อง a และ b
        # (x1, y1) คือพิกัดมุมบนซ้าย, (x2, y2) คือพิกัดมุมล่างขวา
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        #1. คำนวณหาพื้นที่ส่วนที่ซ้อนทับกัน (Intersection) ---
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        # --- 2. คำนวณหาพื้นที่ของทั้งสองกล่องรวมกัน (Union) ---
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    while not _STOP.is_set():
        #อ่านเฟรมล่าสุดจาก Reader
        item = reader.read()
        if item is None:
            time.sleep(0.01); continue

        frame, frame_ts = item
        h, w = frame.shape[:2]
        #ตัดสินใจว่าจะทำ detection ในเฟรมนี้หรือไม่ (ตามค่า DETECT_EVERY_N_FRAMES)
        do_detect = (frame_idx % DETECT_EVERY_N_FRAMES == 0)

        if do_detect:
            #ส่งเฟรมเข้าโมเดล YOLO เพื่อหา object
            try:
                results = model.predict(source=frame, conf=YOLO_CONF, verbose=False, imgsz=1280)
                boxes = results[0].boxes
            except Exception as e:
                print("[YOLO] inference error:", e)
                boxes = None

            if boxes is not None and len(boxes) > 0:
                dets_xyxy = []
                labels = []
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0]); conf = float(box.conf[0])
                        label = model.names[cls_id]
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        dets_xyxy.append([x1, y1, x2, y2, conf])
                        labels.append(label)
                    except Exception:
                        continue
                last_detections_xyxy = np.array(dets_xyxy, dtype=float) if dets_xyxy else np.empty((0,5), dtype=float)
                last_labels = labels
            else:
                last_detections_xyxy = np.empty((0,5), dtype=float)
                last_labels = []
        #ส่งผลลัพธ์ (ไม่ว่าจะมาจากเฟรมปัจจุบันหรือเฟรมก่อนหน้า) เข้า SORT Tracker
        tracker.max_age = TRACK_MAX_AGE
        tracker.min_hits = TRACK_MIN_HITS
        tracker.iou_threshold = TRACK_IOU_THRESH
        tracks = tracker.update(last_detections_xyxy.copy())

        overlay_dets = []
        interesting_to_send = []
        found_labels_txt = []
        now_time = time.time()

        det_boxes = [d[:4] for d in last_detections_xyxy.tolist()]
        det_confs = [d[4] for d in last_detections_xyxy.tolist()]

        # ฟังก์ชันหาค่า IOU (Intersection over Union) เพื่อจับคู่ track กับ detection
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = iw * ih
            area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
            area_b = max(0, bx2-bx1) * max(0, by2-by1)
            union = area_a + area_b - inter + 1e-6
            return inter / union
        #วนลูปผลลัพธ์จาก Tracker
        for t in tracks:
            x1, y1, x2, y2, tid = t.tolist()
            label = "object"; conf = 0.0
            #จับคู่ Track ID กับ Label/Conf จาก YOLO โดยใช้ค่า IoU ที่ดีที่สุด
            if det_boxes:
                ious = [_iou([x1,y1,x2,y2], db) for db in det_boxes]
                j = int(np.argmax(ious)) if len(ious) else -1
                if j >= 0 and ious[j] > 0.1:
                    label = last_labels[j] if j < len(last_labels) else "object"
                    conf = float(det_confs[j] if j < len(det_confs) else 0.0)
            # เตรียมข้อมูลสำหรับวาดกรอบ
            overlay_dets.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "label": label, "conf": conf, "track_id": int(tid),
            })
            found_labels_txt.append(f"{label}#{int(tid)} ({conf:.2f})")

            #ถ้าเป็น object ที่เราสนใจ (และเป็นเฟรมที่ทำ detection) ให้ส่งเข้า Gate
            if do_detect and label in INTEREST_LABELS:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                should_send, track_info, is_stationary, moved_norm = move_gate.update_and_should_send(
                    label, cx, cy, w, h, now=now_time
                )
                if not should_send:
                    continue

                allow, cd = alert_gate.allow(label, cx, cy, w, h, is_stationary, now=now_time)
                if not allow:
                    continue

                interesting_to_send.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "label": label, "conf": conf,
                    "track_id": int(tid),
                    "is_stationary": is_stationary, "moved_norm": moved_norm
                })
        #วาดผลลัพธ์ทั้งหมดลงบนเฟรม
        det_overlay.update(overlay_dets)
        det_overlay.draw(frame)

        if SHOW_WINDOW:
            cv2.imshow("RTSP Stream + SORT tracking (10 fps)", frame)
            key=cv2.waitKey(1) & 0xFF
            if key==ord('q'): break
        #ถ้ามี object ที่น่าสนใจ (ผ่าน Gate) ให้ทำการบันทึกและอัปโหลด
        if interesting_to_send:
            all_visible_interesting_objects = [
                det for det in overlay_dets if det.get("label") in INTEREST_LABELS
            ]
            if not all_visible_interesting_objects:
                all_visible_interesting_objects = interesting_to_send
                
            # สร้างชื่อไฟล์และ path
            dest_dir = get_windows_sd_dir()
            try: os.makedirs(dest_dir, exist_ok=True)
            except Exception: pass

            stamp = datetime.fromtimestamp(frame_ts, tz=LOCAL_TZ)
            labels_summary = "-".join(sorted(list(set(p['label'] for p in all_visible_interesting_objects))))
            fname = f"detect_{stamp.strftime('%Y%m%d_%H%M%S')}_{labels_summary}_{uuid.uuid4().hex[:4]}.jpg"
            filename = os.path.join(dest_dir, fname)
            
            # บันทึกไฟล์ภาพ
            try:
                ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    with open(filename, "wb") as f: f.write(enc.tobytes())
                    _prune_if_needed(dest_dir)
                    threaded_upload(filename, all_visible_interesting_objects, captured_ts=frame_ts)
                else:
                    rgb.red()
                    print("[SAVE] imencode failed")
            except Exception as e:
                rgb.red()
                print(f"[SAVE] write failed:", e)

        stamp_txt = datetime.fromtimestamp(frame_ts, tz=LOCAL_TZ).strftime('%d-%m-%Y_%H:%M:%S %Z')
        print("[TRACK", stamp_txt, "]", ", ".join(found_labels_txt) if found_labels_txt else "-")
        has_internet()

        now=time.time()
        dt=now-last_time
        if dt<frame_period: time.sleep(frame_period-dt)
        last_time=time.time()
        frame_idx += 1

    reader.release()
    if SHOW_WINDOW: cv2.destroyAllWindows()

# --- โหมด error-hold: ไฟแดงค้างและค้างอยู่จนกด Ctrl+C ---
def _fatal_red_halt(msg=""):
    if msg:
        print("[FATAL]", msg)
    try:
        rgb.red()  # ไฟแดงค้าง
    except Exception:
        pass
    print("[LED] Red ON - halting here. Press Ctrl+C to exit.")
    while not _STOP.is_set():
        time.sleep(0.5)

def _cleanup_and_exit():
    try:
        rgb.cleanup()
    finally:
        print("[EXIT] Cleaning up and exiting.")

def _sig_handler(signum, frame):
    _STOP.set()

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def _main():
    # เช็คอินเทอร์เน็ตก่อนเริ่มทั้งหมด
    if not has_internet():
        _fatal_red_halt("[NET] No internet connectivity detected.")
        return

    try:
        t=threading.Thread(target=run_detection_loop, daemon=True); t.start()
        while t.is_alive(): t.join(timeout=0.2)
    finally:
        _cleanup_and_exit()

if __name__ == "__main__":
    _main()
