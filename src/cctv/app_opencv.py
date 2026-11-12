"""
CCTV 헬멧 감지 — 헤드리스 OpenCV 러너
=======================================
GUI 창이 없는 서버(WSL2)용. 결과를 동시에 두 가지로 출력:
  1) MJPEG 실시간 스트림  →  브라우저에서 http://<서버IP>:8000/
  2) 주석 MP4 저장        →  cctv_output/cctv_<시각>.mp4

추가:
  - --grid : 2x2 관제실 멀티캠 그리드 (한 소스를 4개 가상 캠으로)
  - 지속 이벤트 자동 캡처 : 10초+ 지속 시 원본 프레임을 cctv_captures/ 에 저장

입력 소스(--source):
  - (기본) data/data_final_backup/test/images 를 프레임처럼 순차 재생 (가상 CCTV)
  - 동영상 파일 경로 / rtsp://... / 이미지 폴더 경로 / 웹캠 인덱스(정수)

예)
  python src/cctv/app_opencv.py
  python src/cctv/app_opencv.py --grid
  python src/cctv/app_opencv.py --source rtsp://... --no-save
  python src/cctv/app_opencv.py --capture-seconds 3 --max-frames 300
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# 같은 폴더(src/cctv)의 common.py 를 임포트 (리팩토링: cctv_common → src/cctv/common)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cc

DEFAULT_SOURCE = str(cc.PROJECT_DIR / "data/data_final_backup/test/images")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── 프레임 소스 ────────────────────────────────────────────
class FrameSource:
    """동영상/RTSP/웹캠/이미지폴더를 통일된 read() 인터페이스로 제공."""

    def __init__(self, source: str, fps: float = 15.0, shuffle_seed: int | None = None,
                 start_offset: int = 0):
        self.fps = fps
        self.kind = None
        self.cap = None
        self.images: list[Path] = []
        self.idx = 0

        p = Path(source)
        if source.isdigit():  # 웹캠 인덱스
            self.kind = "cap"
            self.cap = cv2.VideoCapture(int(source))
        elif source.startswith("rtsp://") or source.startswith("http"):
            self.kind = "cap"
            self.cap = cv2.VideoCapture(source)
        elif p.is_dir():  # 이미지 폴더 → 가상 CCTV
            self.kind = "images"
            self.images = sorted([q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS])
            if shuffle_seed is not None:
                import random
                random.Random(shuffle_seed).shuffle(self.images)
            if self.images:
                self.idx = start_offset % len(self.images)
        elif p.is_file():  # 동영상 파일
            self.kind = "cap"
            self.cap = cv2.VideoCapture(str(p))
        else:
            raise FileNotFoundError(f"소스를 찾을 수 없음: {source}")

        if self.kind == "images" and not self.images:
            raise FileNotFoundError(f"이미지가 없는 폴더: {source}")

    def read(self):
        if self.kind == "images":
            img = cv2.imread(str(self.images[self.idx]))
            self.idx = (self.idx + 1) % len(self.images)  # 루프
            return img is not None, img
        ok, frame = self.cap.read()
        if not ok and self.kind == "cap":  # 동영상 끝 → 처음으로 루프
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ── MJPEG 스트리밍 서버 (Flask) ────────────────────────────
class MJPEGServer:
    def __init__(self, host="0.0.0.0", port=8000):
        from flask import Flask, Response

        self.host, self.port = host, port
        self._frame = None
        self._lock = threading.Lock()
        app = Flask(__name__)

        @app.route("/")
        def index():
            return (
                "<html><head><title>CCTV 헬멧 감지</title>"
                "<style>body{background:#111;margin:0;text-align:center;font-family:sans-serif}"
                "h2{color:#ddd;padding:10px} img{max-width:100%;height:auto}</style></head>"
                "<body><h2>🪖 CCTV 헬멧 감지 — 실시간</h2>"
                "<img src='/video'></body></html>"
            )

        @app.route("/video")
        def video():
            return Response(self._gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        self.app = app

    def _gen(self):
        while True:
            with self._lock:
                buf = self._frame
            if buf is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
            time.sleep(0.03)

    def update(self, frame_bgr):
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._frame = jpg.tobytes()

    def start(self):
        t = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port,
                                        threaded=True, debug=False, use_reloader=False),
            daemon=True,
        )
        t.start()
        print(f"[mjpeg] 스트림 시작 → http://{self.host}:{self.port}/")


# ── 메인 루프 ──────────────────────────────────────────────
def run(args):
    cc.get_model(args.model)  # 미리 로드

    cap_classes = cc.CLASS_NAMES if args.capture_classes == "all" \
        else [c.strip() for c in args.capture_classes.split(",") if c.strip()]

    # 소스 구성 (그리드면 4개 가상 캠)
    if args.grid:
        n_cam = 4
        sources = [
            FrameSource(args.source, args.fps, shuffle_seed=i, start_offset=i * 137)
            for i in range(n_cam)
        ]
        cam_names = [f"CAM {i + 1}" for i in range(n_cam)]
    else:
        sources = [FrameSource(args.source, args.fps)]
        cam_names = ["CAM 1"]

    trackers = [
        cc.EventTracker(
            capture_seconds=args.capture_seconds,
            recapture_seconds=args.recapture_seconds,
            classes=cap_classes,
            save_annotated=args.save_annotated,
        )
        for _ in sources
    ] if not args.no_capture else None

    # MJPEG
    server = None
    if not args.no_stream:
        server = MJPEGServer(port=args.port)
        server.start()

    # MP4 writer (첫 프레임 크기 확정 후 초기화)
    writer = None
    out_path = None
    if not args.no_save:
        cc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_grid" if args.grid else ""
        out_path = cc.OUTPUT_DIR / f"cctv_{stamp}{suffix}.mp4"

    print(f"[run] 소스={args.source}  grid={args.grid}  capture={not args.no_capture}"
          f"  classes={cap_classes}  threshold={args.capture_seconds}s")
    print("[run] 종료: Ctrl+C")

    frame_period = 1.0 / max(1.0, args.fps)
    fps_ema = args.fps
    n = 0
    try:
        while True:
            t_start = time.perf_counter()

            tiles = []
            for i, src in enumerate(sources):
                ok, frame = src.read()
                if not ok or frame is None:
                    tiles.append(None)
                    continue
                annotated, counts, lat, dets = cc.detect_and_annotate(
                    frame, conf=args.conf, iou=args.iou, cam_name=cam_names[i], fps=fps_ema,
                )
                if trackers is not None:
                    trackers[i].update(counts, frame, annotated, cam_names[i], dets)
                tiles.append(annotated)

            display = cc.draw_grid(tiles) if args.grid else (
                tiles[0] if tiles and tiles[0] is not None else None
            )
            if display is None:
                continue

            if server is not None:
                server.update(display)
            if not args.no_save:
                if writer is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
                    print(f"[mp4] 저장 시작 → {out_path}  ({w}x{h} @ {args.fps}fps)")
                writer.write(display)

            n += 1
            if args.max_frames and n >= args.max_frames:
                print(f"[run] max-frames({args.max_frames}) 도달 — 종료")
                break

            # fps 측정 + 페이싱
            dt = time.perf_counter() - t_start
            inst_fps = 1.0 / dt if dt > 0 else args.fps
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            sleep = frame_period - dt
            if sleep > 0:
                time.sleep(sleep)
            if n % 50 == 0:
                cap_total = sum(t.saved_count for t in trackers) if trackers else 0
                print(f"[run] {n} 프레임  ~{fps_ema:.1f}fps  캡처 {cap_total}장")
    except KeyboardInterrupt:
        print("\n[run] 중단됨")
    finally:
        for s in sources:
            s.release()
        if writer is not None:
            writer.release()
            print(f"[mp4] 저장 완료 → {out_path}")
        if trackers is not None:
            total = sum(t.saved_count for t in trackers)
            print(f"[capture] 총 {total}장 저장 → {cc.CAPTURE_DIR}/")
        if server is not None:
            print("[mjpeg] (스트림 종료를 위해 프로세스가 곧 종료됩니다)")


def parse_args():
    ap = argparse.ArgumentParser(description="CCTV 헬멧 감지 — 헤드리스 OpenCV 러너")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="이미지폴더 / 동영상파일 / rtsp:// / 웹캠인덱스 (기본: 테스트 이미지 가상 CCTV)")
    ap.add_argument("--model", default=None, help=f"모델 가중치 (기본: {cc.MODEL_NAME})")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--port", type=int, default=8000, help="MJPEG 포트")
    ap.add_argument("--grid", action="store_true", help="2x2 관제실 멀티캠 그리드")
    ap.add_argument("--max-frames", type=int, default=0, help="N 프레임 후 종료 (0=무한)")
    ap.add_argument("--no-stream", action="store_true", help="MJPEG 스트리밍 끄기")
    ap.add_argument("--no-save", action="store_true", help="MP4 저장 끄기")
    # 캡처
    ap.add_argument("--no-capture", action="store_true", help="지속 이벤트 캡처 끄기")
    ap.add_argument("--capture-seconds", type=float, default=10.0, help="캡처 임계 지속시간(초)")
    ap.add_argument("--recapture-seconds", type=float, default=10.0, help="지속 중 재캡처 주기(초)")
    ap.add_argument("--capture-classes", default=",".join(cc.ALERT_CLASSES),
                    help="캡처 대상 클래스 콤마구분 또는 'all'")
    ap.add_argument("--save-annotated", action="store_true", help="원본 외 주석본도 저장")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
