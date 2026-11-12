"""
CCTV 헬멧 감지 데모 — 공통 모듈
=================================
app_gradio.py / app_opencv.py 가 공유하는 로직.

- 모델 로드 (round7, 1회 캐시)
- detect_and_annotate(): 추론 + 예쁜 OSD 드로잉 (코너 박스/카운트 칩/경고 배너/REC)
- draw_grid(): 2x2 관제실 그리드 합성
- EventTracker: 10초+ 지속 이벤트 자동 캡처 (재학습용 원본 프레임 저장)

클래스: 0=fallen  1=no_helmet  2=ok   (data/data_final_backup/data.yaml)
색상은 OpenCV 기준 BGR.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── 설정 ──────────────────────────────────────────────────
# 이 파일 위치: <root>/src/cctv/common.py → 프로젝트 루트는 3단계 위(parents[2])
PROJECT_DIR = Path(__file__).resolve().parents[2]
# 최종 채택 모델(round7). 제자리 유지 경로(절대경로 결합) — 변경 금지.
MODEL_PATH = str(PROJECT_DIR / "experiments/runs2/round7/weights/best.pt")  # round7
MODEL_NAME = "round7"
IMG_SIZE = 640

# class id -> (name, BGR color, emoji)
CLASS_INFO = {
    0: {"name": "fallen",    "color": (60, 60, 240),  "emoji": "🔴"},   # 빨강
    1: {"name": "no_helmet", "color": (0, 170, 255),  "emoji": "🟡"},   # 주황
    2: {"name": "ok",        "color": (80, 200, 90),  "emoji": "🟢"},   # 초록
}
CLASS_NAMES = [CLASS_INFO[i]["name"] for i in sorted(CLASS_INFO)]
NAME_TO_ID = {v["name"]: k for k, v in CLASS_INFO.items()}
ALERT_CLASSES = ("fallen", "no_helmet")  # 기본 경고/캡처 대상

# 데모 산출물은 demo_outputs/ 로 통합됨 (리팩토링 2026-06)
CAPTURE_DIR = PROJECT_DIR / "demo_outputs" / "cctv_captures"
OUTPUT_DIR = PROJECT_DIR / "demo_outputs" / "cctv_output"

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── 모델 로드 (전역 캐시) ──────────────────────────────────
_model = None
_model_path = None


def get_model(model_path: str | None = None):
    """YOLO 모델을 1회 로드해 캐시. model_path 가 바뀌면 재로드."""
    global _model, _model_path
    path = model_path or MODEL_PATH
    if _model is None or path != _model_path:
        from ultralytics import YOLO  # 지연 import (로드 비용)

        print(f"[cctv] 모델 로드: {path}")
        _model = YOLO(path)
        _model.overrides["verbose"] = False
        _model_path = path
        print("[cctv] 모델 로드 완료 ✓")
    return _model


# ── 드로잉 헬퍼 ────────────────────────────────────────────
def _corner_box(img, x1, y1, x2, y2, color, thick=2):
    """모서리 강조(코너 브래킷) 박스 — CCTV 느낌."""
    w, h = x2 - x1, y2 - y1
    L = max(12, int(min(w, h) * 0.22))
    # 얇은 전체 테두리
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # 굵은 코너
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)


def _label_tag(img, x, y, text, color):
    """박스 좌상단 라벨 태그."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    ytop = max(0, y - th - 9)
    cv2.rectangle(img, (x, ytop), (x + tw + 10, ytop + th + 9), color, -1)
    cv2.putText(img, text, (x + 5, ytop + th + 2), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _alpha_rect(img, p1, p2, color, alpha):
    """반투명 사각형."""
    x1, y1 = p1
    x2, y2 = p2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


# ── 추론 + 주석 ────────────────────────────────────────────
def detect_and_annotate(
    frame_bgr: np.ndarray,
    conf: float = 0.35,
    iou: float = 0.45,
    show_label: bool = True,
    show_conf: bool = True,
    cam_name: str = "CAM 1",
    fps: float | None = None,
    model=None,
):
    """
    BGR 프레임을 받아 추론하고 예쁜 OSD 를 그려 반환.
    returns: (annotated_bgr, counts{name:int}, latency_ms, dets[(name,conf)])
    """
    model = model or get_model()
    t0 = time.perf_counter()
    results = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out = frame_bgr.copy()
    res = results[0]
    counts = {name: 0 for name in CLASS_NAMES}
    dets: list[tuple[str, float]] = []

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        clss = res.boxes.cls.cpu().numpy().astype(int)
        cfs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cls, c in zip(xyxy, clss, cfs):
            info = CLASS_INFO.get(int(cls), {"name": str(cls), "color": (200, 200, 200)})
            name = info["name"]
            counts[name] = counts.get(name, 0) + 1
            dets.append((name, float(c)))
            _corner_box(out, x1, y1, x2, y2, info["color"])
            if show_label:
                txt = f"{name} {c:.2f}" if show_conf else name
                _label_tag(out, x1, y1, txt, info["color"])

    _draw_osd(out, counts, latency_ms, cam_name, fps)
    return out, counts, latency_ms, dets


def _draw_osd(out, counts, latency_ms, cam_name, fps):
    H, W = out.shape[:2]

    # 상단 바 (반투명)
    _alpha_rect(out, (0, 0), (W, 34), (15, 15, 15), 0.55)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(out, cam_name, (10, 23), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, ts, (W // 2 - 90, 23), _FONT, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    info_r = f"{MODEL_NAME}"
    if fps is not None:
        info_r += f" | {fps:4.1f} FPS"
    info_r += f" | {latency_ms:4.0f}ms"
    (tw, _), _ = cv2.getTextSize(info_r, _FONT, 0.5, 1)
    cv2.putText(out, info_r, (W - tw - 70, 22), _FONT, 0.5, (180, 230, 180), 1, cv2.LINE_AA)
    # ● REC (점멸)
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(out, (W - 48, 17), 7, (60, 60, 240), -1, cv2.LINE_AA)
    cv2.putText(out, "REC", (W - 38, 22), _FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # 우상단 카운트 칩
    chip_y = 44
    for cid in sorted(CLASS_INFO):
        info = CLASS_INFO[cid]
        n = counts.get(info["name"], 0)
        txt = f"{info['name']}: {n}"
        (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.5, 1)
        x2 = W - 10
        x1 = x2 - tw - 16
        _alpha_rect(out, (x1, chip_y), (x2, chip_y + th + 12), (25, 25, 25), 0.6)
        cv2.rectangle(out, (x1, chip_y), (x1 + 6, chip_y + th + 12), info["color"], -1)
        cv2.putText(out, txt, (x1 + 12, chip_y + th + 4), _FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        chip_y += th + 18

    # 경고 배너
    alerts = [c for c in ALERT_CLASSES if counts.get(c, 0) > 0]
    if alerts and int(time.time() * 2) % 2 == 0:
        _alpha_rect(out, (0, H - 40), (W, H), (40, 40, 200), 0.55)
        msg = "  ".join(f"! {c.upper()} x{counts[c]}" for c in alerts)
        cv2.putText(out, msg, (14, H - 14), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ── 멀티캠 그리드 ──────────────────────────────────────────
def draw_grid(frames: list[np.ndarray], cell=(640, 360)) -> np.ndarray:
    """프레임 리스트를 2x2(또는 그 이상) 그리드로 합성. 빈 칸은 검정."""
    cw, ch = cell
    n = len(frames)
    cols = 2 if n <= 4 else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    mosaic = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if i < n and frames[i] is not None:
            tile = cv2.resize(frames[i], (cw, ch))
        else:
            tile = np.full((ch, cw, 3), 18, dtype=np.uint8)
            cv2.putText(tile, "NO SIGNAL", (cw // 2 - 70, ch // 2), _FONT, 0.7, (90, 90, 90), 2)
        mosaic[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = tile
        cv2.rectangle(mosaic, (c * cw, r * ch), ((c + 1) * cw - 1, (r + 1) * ch - 1), (60, 60, 60), 1)
    return mosaic


# ── 지속 이벤트 자동 캡처 ──────────────────────────────────
class EventTracker:
    """
    클래스별 연속 감지 시간을 추적해, 임계(기본 10초) 이상 지속되면
    원본 프레임을 cctv_captures/<class>/ 에 저장 (재학습용 데이터).

    - grace: 짧게 사라졌다 다시 잡히면 타이머 유지 (깜빡임 무시)
    - recapture_seconds: 이벤트 지속 중 재캡처 주기
    """

    def __init__(
        self,
        capture_seconds: float = 10.0,
        recapture_seconds: float = 10.0,
        grace: float = 1.0,
        classes=ALERT_CLASSES,
        save_annotated: bool = False,
        out_dir: Path | str = CAPTURE_DIR,
    ):
        self.capture_seconds = capture_seconds
        self.recapture_seconds = recapture_seconds
        self.grace = grace
        self.classes = tuple(CLASS_NAMES) if classes in ("all", ("all",)) else tuple(classes)
        self.save_annotated = save_annotated
        self.out_dir = Path(out_dir)
        self.csv_path = self.out_dir / "events.csv"
        self.saved_count = 0
        # cls -> {start, last_seen, last_capture}
        self._state: dict[str, dict] = {}
        self._ensure_csv()

    def _ensure_csv(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["time", "cam", "class", "duration_s", "conf", "filepath"]
                )

    def update(self, counts, raw_frame, annotated_frame=None, cam_name="CAM 1", dets=None):
        """매 프레임 호출. 이번 호출에서 저장한 파일 수를 반환."""
        now = time.monotonic()
        # 클래스별 최대 신뢰도
        max_conf = {}
        for name, c in (dets or []):
            if c > max_conf.get(name, 0):
                max_conf[name] = c

        saved = 0
        for cls in self.classes:
            present = counts.get(cls, 0) > 0
            st = self._state.get(cls)

            if present:
                if st is None or (now - st["last_seen"]) > self.grace:
                    st = {"start": now, "last_seen": now, "last_capture": None}
                    self._state[cls] = st
                st["last_seen"] = now
                duration = now - st["start"]
                if duration >= self.capture_seconds:
                    lc = st["last_capture"]
                    if lc is None or (now - lc) >= self.recapture_seconds:
                        self._save(cls, raw_frame, annotated_frame, cam_name,
                                   duration, max_conf.get(cls, 0.0))
                        st["last_capture"] = now
                        saved += 1
            else:
                if st is not None and (now - st["last_seen"]) > self.grace:
                    self._state.pop(cls, None)

        self.saved_count += saved
        return saved

    def _save(self, cls, raw_frame, annotated_frame, cam_name, duration, conf):
        cam_tag = cam_name.replace(" ", "").replace("/", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"
        cls_dir = self.out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{cam_tag}_{stamp}_conf{conf:.2f}.jpg"
        fpath = cls_dir / fname
        cv2.imwrite(str(fpath), raw_frame)  # 박스 없는 원본 (재학습용)
        if self.save_annotated and annotated_frame is not None:
            cv2.imwrite(str(cls_dir / f"annotated_{fname}"), annotated_frame)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().isoformat(timespec="seconds"), cam_name, cls,
                 f"{duration:.1f}", f"{conf:.2f}", str(fpath)]
            )
        print(f"[capture] {cls} {duration:.0f}s 지속 → {fpath}")
