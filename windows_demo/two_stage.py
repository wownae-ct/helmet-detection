"""
2-stage 헬멧 감지 (Windows 이식 버전 — 상대경로)
Stage-1 COCO yolov8s 'person' 검출 / Stage-2 머리크롭 분류(ok/no_helmet) / fallen=round7
모델은 ./models/ 에서 로드. 이 폴더째로 복사하면 어디서든 동작.
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
PERSON_MODEL = str(HERE / 'models' / 'yolov8s.pt')      # COCO pretrained
CLS_MODEL    = str(HERE / 'models' / 'helmet_cls.pt')   # 머리크롭 분류기
FALLEN_MODEL = str(HERE / 'models' / 'fallen_round7.pt')# 0=fallen

COLORS = {'ok': (0, 200, 0), 'no_helmet': (0, 0, 230), 'fallen': (0, 140, 255)}
HEAD_H = 0.16        # person 박스 상단 16% ≈ 머리
HEAD_W = 0.70
OK_THRESH = 0.90     # 안전 편향: P(ok)≥0.90 일 때만 ok, 아니면 no_helmet(위반)

_models = {}
def _load():
    if not _models:
        _models['person'] = YOLO(PERSON_MODEL)
        _models['cls']    = YOLO(CLS_MODEL)
        _models['fallen'] = YOLO(FALLEN_MODEL)
    return _models

def head_region(x1, y1, x2, y2):
    w = x2 - x1
    return (int(x1 + w*(1-HEAD_W)/2), int(y1),
            int(x2 - w*(1-HEAD_W)/2), int(y1 + (y2-y1)*HEAD_H))

def detect_two_stage(frame_bgr, person_conf=0.30, fallen_conf=0.45):
    M = _load(); H, W = frame_bgr.shape[:2]; dets = []
    pr = M['person'].predict(frame_bgr, classes=[0], conf=person_conf, verbose=False)[0]
    crops, metas = [], []
    for b in pr.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        hx1, hy1, hx2, hy2 = head_region(x1, y1, x2, y2)
        hx1, hy1 = max(0, hx1), max(0, hy1); hx2, hy2 = min(W, hx2), min(H, hy2)
        if hx2-hx1 < 8 or hy2-hy1 < 8: continue
        crops.append(frame_bgr[hy1:hy2, hx1:hx2]); metas.append((x1, y1, x2, y2))
    if crops:
        cr = M['cls'].predict(crops, imgsz=128, verbose=False)
        names = cr[0].names; ok_i = [k for k, v in names.items() if v == 'ok'][0]
        for meta, c in zip(metas, cr):
            p_ok = float(c.probs.data[ok_i])
            if p_ok >= OK_THRESH: label, conf = 'ok', p_ok
            else: label, conf = 'no_helmet', 1 - p_ok
            dets.append({'box': meta, 'label': label, 'conf': conf})
    fr = M['fallen'].predict(frame_bgr, classes=[0], conf=fallen_conf, verbose=False)[0]
    for b in fr.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        dets.append({'box': (x1, y1, x2, y2), 'label': 'fallen', 'conf': float(b.conf[0])})
    return dets

def draw(frame_bgr, dets):
    out = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d['box']; col = COLORS.get(d['label'], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d['label']} {d['conf']:.2f}", (x1, max(12, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image'); ap.add_argument('--video'); ap.add_argument('--out', default='out.jpg')
    a = ap.parse_args()
    if a.image: fr = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.video); n = int(cap.get(7)); cap.set(1, n//2)
        ok, fr = cap.read(); cap.release()
    cv2.imwrite(a.out, draw(fr, detect_two_stage(fr))); print("저장:", a.out)

if __name__ == '__main__':
    main()
