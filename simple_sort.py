# simple_sort.py  —  Pure-Python SORT (Kalman + Hungarian), no C/C++ build required.
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    # bb_test: Mx4, bb_gt: Nx4  -> IoU matrix MxN
    M = bb_test.shape[0]
    N = bb_gt.shape[0]
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)
    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt   = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_test[:, None] + area_gt[None, :] - inter
    return inter / (union + 1e-6)

def convert_bbox_to_z(bbox):
    # แปลงพิกัด [x1,y1,x2,y2] (มุมซ้ายบน, มุมขวาล่าง) ให้อยู่ในรูปแบบที่ Kalman Filter ใช้
    #คือ [cx, cy, s, r] โดย cx,cy=จุดศูนย์กลาง, s=พื้นที่, r=อัตราส่วนกว้างยาว
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    # แปลงค่า state จาก Kalman Filter [cx, cy, s, r, ...] กลับไปเป็น [x1,y1,x2,y2]
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker:
    #Class นี้คือ Tracker สำหรับวัตถุ "แต่ละชิ้น"
    #ภายในจะใช้ Kalman Filter เพื่อทำนายตำแหน่งของวัตถุในเฟรมถัดไป
    #และอัปเดตตำแหน่งเมื่อได้รับข้อมูลใหม่ (detection)
    count = 0
    def __init__(self, bbox):
        # 7-dim state: [cx, cy, s, r, vx, vy, vs]; r (aspect) ไม่มีความเร็ว
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # F คือ State Transition Matrix: ใช้ในการทำนาย state ใน step ถัดไป
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], dtype=np.float32)
        # H คือ Measurement Function: ใช้แปลง state ให้อยู่ในรูปของ measurement ([cx,cy,s,r])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ], dtype=np.float32)
        self.kf.R[2:,2:] *= 10.0
        self.kf.P[4:,4:] *= 1000.0  # high uncertainty for velocities
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # ตัวแปรสำหรับจัดการอายุและการมองเห็นของ tracker
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.kf.x[:4] = convert_bbox_to_z(bbox)

    def update(self, bbox):
        # อัปเดต state ของ Kalman Filter ด้วย bounding box ที่ตรวจจับได้จริง (measurement)
        #จะถูกเรียกเมื่อ tracker ถูกจับคู่กับ detection ได้สำเร็จ
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        #ทำนายตำแหน่งของ bounding box ในเฟรมถัดไป โดยใช้ state ปัจจุบัน
        #จะถูกเรียกทุกๆ เฟรม ไม่ว่าจะเจอ detection ที่คู่กันหรือไม่ก็ตาม
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        #ดึงค่าตำแหน่งปัจจุบันของ tracker ออกมาในรูปแบบ [x1,y1,x2,y2]
        return convert_x_to_bbox(self.kf.x)

class Sort:
    #Class หลักของ SORT algorithm ที่จัดการ tracker ทั้งหมด
    #ทำหน้าที่รับ detections จากโมเดลในแต่ละเฟรม แล้วจับคู่กับ tracker ที่มีอยู่,
    #สร้าง tracker ใหม่สำหรับวัตถุใหม่, และลบ tracker ที่เก่าเกินไป
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.2):
        self.max_age = max_age # อายุสูงสุด (เฟรม) ที่ tracker จะอยู่ได้ถ้าไม่เจอคู่
        self.min_hits = min_hits # จำนวนครั้งที่ต้องเจอติดต่อกันเพื่อเริ่มแสดงผล
        self.iou_threshold = iou_threshold # ค่า IoU ขั้นต่ำที่จะถือว่าเป็นการจับคู่ที่ถูกต้อง
        self.trackers = []

    def update(self, dets=np.empty((0,5))):
        #รับ detections ใหม่ (dets) และอัปเดตสถานะของ tracker ทั้งหมด
        # dets: Nx5 -> [x1 y1 x2 y2 score]
        ret = []
        # --- 1. ทำนายตำแหน่งใหม่ของ tracker ที่มีอยู่ทั้งหมด ---
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        self.trackers = [t for i,t in enumerate(self.trackers) if i not in to_del]

        # --- 2. จับคู่ detections ใหม่ กับ tracker ที่ทำนายตำแหน่งไว้ โดยใช้ Hungarian Algorithm ---
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # --- 3. อัปเดต tracker ที่หาคู่เจอ (matched) ด้วยข้อมูล detection ใหม่ ---
        for m in matched:
            d = dets[m[0], :4]
            self.trackers[m[1]].update(d)

        # --- 4. สร้าง tracker ใหม่สำหรับ detection ที่หาคู่ไม่เจอ (unmatched_dets) ---
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # --- 5. รวบรวมผลลัพธ์ และลบ tracker ที่เก่าหรือหายไปนานเกินไป (max_age) ---
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # เงื่อนไขการแสดงผล: ต้องเป็น tracker ที่เพิ่งอัปเดต และมีความน่าเชื่อถือ (ผ่าน min_hits)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or trk.age <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1,-1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) == 0:
            return np.empty((0,5))
        return np.concatenate(ret)

    def _associate_detections_to_trackers(self, dets, trks):
        #ฟังก์ชันสำหรับจับคู่ Detections กับ Trackers
        #1. คำนวณค่า IoU ระหว่างทุก detection กับทุก tracker เพื่อสร้าง Cost Matrix
        #2. ใช้ Hungarian algorithm (linear_sum_assignment) เพื่อหาการจับคู่ที่ดีที่สุด
        #  ที่ทำให้ผลรวมของ IoU สูงที่สุด
        #3. คืนค่า index ของคู่ที่จับคู่ได้, detection ที่ไม่มีคู่, และ tracker ที่ไม่มีคู่
        if trks.shape[0] == 0 or dets.shape[0] == 0:
            return np.empty((0,2), dtype=int), np.arange(dets.shape[0]), np.arange(trks.shape[0])

        iou_mat = iou_batch(dets[:, :4], trks)
        # เราใส่เครื่องหมายลบ เพราะ linear_sum_assignment จะหาค่าผลรวมที่ "น้อยที่สุด"
        # แต่เราต้องการหาผลรวม IoU ที่ "มากที่สุด"
        row_ind, col_ind = linear_sum_assignment(-iou_mat)  # maximize IoU
        matches = []
        # กรองคู่ที่ค่า IoU ต่ำกว่า threshold ออกไป
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                matches.append([r, c])
        matches = np.array(matches, dtype=int)

        # หา index ของ detections และ trackers ที่ไม่มีคู่
        matched_dets = matches[:, 0] if matches.size else np.array([], dtype=int)
        matched_trks = matches[:, 1] if matches.size else np.array([], dtype=int)
        unmatched_dets = np.setdiff1d(np.arange(dets.shape[0]), matched_dets)
        unmatched_trks = np.setdiff1d(np.arange(trks.shape[0]), matched_trks)
        return matches, unmatched_dets, unmatched_trks
