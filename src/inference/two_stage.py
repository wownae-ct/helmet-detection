"""
2-stage 헬멧 감지 추론
======================
Stage-1  COCO yolov8s 'person' 검출 (도메인 강건 — 가까운 얼굴까지 일반화)
Stage-2  각 person 머리영역 크롭 → helmet_cls 분류 → ok / no_helmet
fallen   기존 검출 모델(round7)로 fallen 만 병행 검출

사용:
  python src/inference/two_stage.py --video cctv_output/xxx.mp4 --out /tmp/xx_2stage.jpg
  (또는 detect_two_stage(frame_bgr) 를 src/cctv/common 에서 호출)
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path('/mnt/ml/helmet_project')
PERSON_MODEL = 'yolov8s.pt'                                   # COCO pretrained
CLS_MODEL    = str(BASE / 'runs/classify/experiments/helmet_cls/weights/best.pt')
FALLEN_MODEL = str(BASE / 'experiments/runs2/round7/weights/best.pt')  # 0=fallen

COLORS = {'ok': (0, 200, 0), 'no_helmet': (0, 0, 230), 'fallen': (0, 140, 255)}
HEAD_H = 0.16   # person 박스 상단 16% ≈ 머리(서있는 사람 ~7두신)
HEAD_W = 0.70   # 중앙 70% 폭
OK_THRESH = 0.90  # 안전 편향: P(ok)≥0.90 일 때만 'ok', 아니면 no_helmet(위반)
                  #   캡·애매한 머리는 no_helmet(위반)으로 → 놓침 방지. 하드햇만 ok.

_models = {}
def _load():
    if not _models:
        _models['person'] = YOLO(PERSON_MODEL)
        _models['cls']    = YOLO(CLS_MODEL)
        _models['fallen'] = YOLO(FALLEN_MODEL)
    return _models


def head_region(x1, y1, x2, y2):
    w = x2 - x1
    hx1 = int(x1 + w * (1 - HEAD_W) / 2)
    hx2 = int(x2 - w * (1 - HEAD_W) / 2)
    hy2 = int(y1 + (y2 - y1) * HEAD_H)
    return hx1, int(y1), hx2, hy2


def detect_two_stage(frame_bgr, person_conf=0.30, fallen_conf=0.45):
    """returns list of dicts: {box:(x1,y1,x2,y2), label, conf}"""
    M = _load()
    H, W = frame_bgr.shape[:2]
    dets = []

    # Stage-1: person
    pr = M['person'].predict(frame_bgr, classes=[0], conf=person_conf, verbose=False)[0]
    crops, metas = [], []
    for b in pr.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        hx1, hy1, hx2, hy2 = head_region(x1, y1, x2, y2)
        hx1, hy1 = max(0, hx1), max(0, hy1); hx2, hy2 = min(W, hx2), min(H, hy2)
        if hx2 - hx1 < 8 or hy2 - hy1 < 8:
            continue
        crops.append(frame_bgr[hy1:hy2, hx1:hx2])
        metas.append((x1, y1, x2, y2))

    # Stage-2: classify head crops (batch)
    if crops:
        cr = M['cls'].predict(crops, imgsz=128, verbose=False)
        names = cr[0].names
        ok_i = [k for k, v in names.items() if v == 'ok'][0]
        for meta, c in zip(metas, cr):
            p_ok = float(c.probs.data[ok_i])
            # 안전 편향: 확실한 헬멧만 ok, 나머지(캡·애매)는 no_helmet
            if p_ok >= OK_THRESH:
                label, conf = 'ok', p_ok
            else:
                label, conf = 'no_helmet', 1 - p_ok
            dets.append({'box': meta, 'label': label, 'conf': conf})

    # fallen (기존 검출기)
    fr = M['fallen'].predict(frame_bgr, classes=[0], conf=fallen_conf, verbose=False)[0]
    for b in fr.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        dets.append({'box': (x1, y1, x2, y2), 'label': 'fallen', 'conf': float(b.conf[0])})
    return dets


def draw(frame_bgr, dets):
    out = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d['box']
        col = COLORS.get(d['label'], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = f"{d['label']} {d['conf']:.2f}"
        cv2.putText(out, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video'); ap.add_argument('--image')
    ap.add_argument('--out', default='/tmp/twostage_out.jpg')
    ap.add_argument('--frame', type=int, default=-1)
    a = ap.parse_args()
    if a.image:
        fr = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame if a.frame >= 0 else n // 2)
        ok, fr = cap.read(); cap.release()
        if not ok: raise SystemExit("프레임 읽기 실패")
    dets = detect_two_stage(fr)
    cv2.imwrite(a.out, draw(fr, dets))
    from collections import Counter
    print("검출:", dict(Counter(d['label'] for d in dets)), "→", a.out)


if __name__ == '__main__':
    main()
