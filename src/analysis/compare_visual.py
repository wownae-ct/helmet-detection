"""
[2] round6 vs round7 '시각' 비교 (라벨 불필요)
================================================
같은 입력(이미지/폴더/영상/웹캠)에 두 모델 추론을 '나란히' 그려 저장하고,
클래스별 검출 수를 카운트한다. → ok/no_helmet 구분 차이를 눈으로 확인.
왼쪽=round6, 오른쪽=round7.  빨강=no_helmet · 초록=ok · 주황=fallen.

[실행]
  python src/analysis/compare_visual.py --source 영상.mp4 --out cmp.mp4
  python src/analysis/compare_visual.py --source 사진폴더   --out cmp_dir
  python src/analysis/compare_visual.py --source 사진.jpg   --out cmp.jpg
  python src/analysis/compare_visual.py --source 0          --out cmp.mp4   # 웹캠(Windows, 모델 복사 후)
옵션:  --conf 0.25 (낮추면 더 많이 잡음)
"""
import argparse, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from ultralytics import YOLO

NAMES = {0: 'fallen', 1: 'no_helmet', 2: 'ok'}
COLORS = {'fallen': (0, 140, 255), 'no_helmet': (0, 0, 230), 'ok': (0, 200, 0)}
DEF6 = "/mnt/ml/helmet_project/experiments/runs2/round6/weights/best.pt"
DEF7 = "/mnt/ml/helmet_project/experiments/runs2/round7/weights/best.pt"
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
VID_EXT = {'.mp4', '.avi', '.mov', '.mkv'}


def draw(frame, result, title, counts):
    out = frame.copy()
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        ci = int(b.cls[0]); name = NAMES.get(ci, str(ci)); conf = float(b.conf[0])
        counts[name] = counts.get(name, 0) + 1
        col = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{name} {conf:.2f}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def pair(m6, m7, frame, conf, c6, c7):
    r6 = m6.predict(frame, conf=conf, verbose=False)[0]
    r7 = m7.predict(frame, conf=conf, verbose=False)[0]
    return np.hstack([draw(frame, r6, "round6", c6), draw(frame, r7, "round7", c7)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', default='compare_out')
    ap.add_argument('--model6', default=DEF6)
    ap.add_argument('--model7', default=DEF7)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()

    m6, m7 = YOLO(a.model6), YOLO(a.model7)
    c6, c7 = {}, {}
    src = a.source
    p = Path(src)

    if src.isdigit() or p.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise SystemExit(f"열기 실패: {src}")
        w, h = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or 20
        out_path = a.out if a.out.lower().endswith(tuple(VID_EXT)) else a.out + '.mp4'
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            vw.write(pair(m6, m7, fr, a.conf, c6, c7)); n += 1
        cap.release(); vw.release()
        print(f"저장: {out_path}  ({n}프레임, 좌=round6 / 우=round7)")
    elif p.is_dir():
        os.makedirs(a.out, exist_ok=True)
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXT])
        for f in imgs:
            fr = cv2.imread(str(f))
            if fr is None:
                continue
            cv2.imwrite(os.path.join(a.out, f.name), pair(m6, m7, fr, a.conf, c6, c7))
        print(f"저장: {a.out}/  ({len(imgs)}장)")
    else:
        fr = cv2.imread(src)
        if fr is None:
            raise SystemExit(f"이미지 못 읽음: {src}")
        out_path = a.out if Path(a.out).suffix else a.out + '.jpg'
        cv2.imwrite(out_path, pair(m6, m7, fr, a.conf, c6, c7))
        print(f"저장: {out_path}")

    print(f"\n=== 클래스별 '검출 수' 비교 (라벨 없음, 참고용 / conf={a.conf}) ===")
    print(f"{'class':12}{'round6':>9}{'round7':>9}")
    for nm in ('fallen', 'no_helmet', 'ok'):
        print(f"{nm:12}{c6.get(nm, 0):>9}{c7.get(nm, 0):>9}")
    print("→ 같은 영상에서 두 모델이 ok/no_helmet을 각각 몇 번 잡았는지. 좌우 영상으로 오분류도 눈으로 확인하세요.")


if __name__ == '__main__':
    main()
