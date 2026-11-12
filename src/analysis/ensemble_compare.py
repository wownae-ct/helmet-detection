"""
[C] 앙상블 비교 — round6 / round7 / (round11) 단일 vs round6+round7 WBF 앙상블
=========================================================================
같은 test셋에서 동일 기준(conf 0.25, IoU 0.5 매칭)으로:
  · 클래스별 precision / recall / F1
  · 안전 핵심: no_helmet→ok 오인(위험), ok→no_helmet 오인(오경보)
앙상블이 단일 모델 대비 의미있게 향상되는지 → "모델 2개 운영할 가치" 판단.

실행:  python src/analysis/ensemble_compare.py
       python src/analysis/ensemble_compare.py --extra round11=experiments/runs2/round11/weights/best.pt
"""
import argparse, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

NAMES = {0: 'fallen', 1: 'no_helmet', 2: 'ok'}
IMGDIR = Path("/mnt/ml/helmet_project/data/data_final_backup/test/images")
LBLDIR = Path("/mnt/ml/helmet_project/data/data_final_backup/test/labels")
R6 = "/mnt/ml/helmet_project/experiments/runs2/round6/weights/best.pt"
R7 = "/mnt/ml/helmet_project/experiments/runs2/round7/weights/best.pt"
CONF, IOU_MATCH = 0.25, 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_gt(stem, W, H):
    f = LBLDIR / f"{stem}.txt"
    gts = []
    if f.exists():
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                gts.append((c, (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H))
    return gts


def predict(model, img):
    r = model.predict(str(img), conf=0.001, iou=0.6, verbose=False)[0]
    H, W = r.orig_shape
    boxes = [(int(b.cls[0]), float(b.conf[0]), *map(float, b.xyxy[0].tolist())) for b in r.boxes]
    return boxes, W, H


def fuse(pred_lists, W, H):
    bl, sl, ll = [], [], []
    for preds in pred_lists:
        bl.append([[max(0,min(1,x1/W)), max(0,min(1,y1/H)), max(0,min(1,x2/W)), max(0,min(1,y2/H))]
                   for (_, _, x1, y1, x2, y2) in preds])
        sl.append([c for (_, c, *_) in preds])
        ll.append([cl for (cl, *_) in preds])
    if not any(bl):
        return []
    b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=[1, 1], iou_thr=0.55, skip_box_thr=0.001)
    return [(int(l[i]), float(s[i]), b[i][0]*W, b[i][1]*H, b[i][2]*W, b[i][3]*H) for i in range(len(b))]


def evaluate(per_image):
    """per_image: list of (gts, dets).  dets=(cls,conf,x1,y1,x2,y2)"""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    nh_to_ok = 0; ok_to_nh = 0
    for gts, dets in per_image:
        dets = [d for d in dets if d[1] >= CONF]
        dets.sort(key=lambda d: -d[1])
        used = [False]*len(gts)
        for d in dets:
            best, bi = 0, -1
            for i, g in enumerate(gts):
                if used[i] or g[0] != d[0]:
                    continue
                v = iou(d[2:], g[1:])
                if v >= IOU_MATCH and v > best:
                    best, bi = v, i
            if bi >= 0:
                tp[d[0]] += 1; used[bi] = True
            else:
                fp[d[0]] += 1
        for i, g in enumerate(gts):
            if not used[i]:
                fn[g[0]] += 1
        # 안전 오분류: GT no_helmet 위치에 ok로 예측 / GT ok 위치에 no_helmet 예측
        for g in gts:
            for d in dets:
                if iou(d[2:], g[1:]) >= IOU_MATCH:
                    if g[0] == 1 and d[0] == 2: nh_to_ok += 1
                    if g[0] == 2 and d[0] == 1: ok_to_nh += 1
                    break
    return tp, fp, fn, nh_to_ok, ok_to_nh


def report(name, ev):
    tp, fp, fn, nh_ok, ok_nh = ev
    print(f"\n=== {name} ===")
    print(f"{'클래스':12}{'P':>8}{'R':>8}{'F1':>8}{'TP/FP/FN':>16}")
    for c in (0, 1, 2):
        p = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
        r = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        print(f"{NAMES[c]:12}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{f'{tp[c]}/{fp[c]}/{fn[c]}':>16}")
    print(f"  🔴 no_helmet→ok 오인(위험): {nh_ok}    🟡 ok→no_helmet(오경보): {ok_nh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extra', action='append', default=[], help='name=path (예: round11=.../best.pt)')
    a = ap.parse_args()
    imgs = sorted([p for p in IMGDIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    print(f"test 이미지: {len(imgs)}장  (conf≥{CONF}, IoU매칭 {IOU_MATCH})")

    m6, m7 = YOLO(R6), YOLO(R7)
    extras = {}
    for e in a.extra:
        nm, pth = e.split('=', 1)
        if Path(pth).exists():
            extras[nm] = YOLO(pth)

    pi6, pi7, piE = [], [], []
    pix = {nm: [] for nm in extras}
    for img in imgs:
        b6, W, H = predict(m6, img)
        b7, _, _ = predict(m7, img)
        gts = load_gt(img.stem, W, H)
        pi6.append((gts, b6)); pi7.append((gts, b7))
        piE.append((gts, fuse([b6, b7], W, H)))
        for nm, mdl in extras.items():
            bx, _, _ = predict(mdl, img)
            pix[nm].append((gts, bx))

    report("round6 (단일)", evaluate(pi6))
    report("round7 (단일)", evaluate(pi7))
    for nm in extras:
        report(f"{nm} (단일)", evaluate(pix[nm]))
    report("round6+round7 WBF 앙상블", evaluate(piE))
    print("\n▶ 앙상블 F1/오분류가 최고 단일 모델보다 '의미있게' 나으면 → 2모델 운영 가치 있음. 미미하면 단일 권장.")


if __name__ == '__main__':
    main()
