"""
CCTV 헬멧 감지 — Gradio 브라우저 웹캠 데모
============================================
브라우저 웹캠을 받아 실시간 감지. 헤드리스 서버(WSL2)에서도 동작
(브라우저가 카메라를 잡아 서버로 프레임 전송).

  실행:  python src/cctv/app_gradio.py
  접속:  http://<서버IP>:7860/   (웹캠 권한 허용)

기능:
  - round7 모델, conf/iou 슬라이더, 라벨·신뢰도 토글
  - 예쁜 OSD (코너 박스/카운트 칩/경고 배너) + 상태 카드
  - 지속 이벤트 자동 캡처(10초+) → demo_outputs/cctv_captures/ (재학습용)
의존: 같은 폴더 common.py(공통 모듈).
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import gradio as gr

# 같은 폴더(src/cctv)의 common.py 를 임포트 (리팩토링: cctv_common → src/cctv/common)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cc

cc.get_model()  # 시작 시 로드

# ── 세션 전역 상태 ─────────────────────────────────────────
_tracker = cc.EventTracker(capture_seconds=10.0, classes=cc.ALERT_CLASSES)
_last_ts = {"t": None}


def _fps():
    now = time.perf_counter()
    prev = _last_ts["t"]
    _last_ts["t"] = now
    if prev is None:
        return 0.0
    dt = now - prev
    return 1.0 / dt if dt > 0 else 0.0


def _status_html(counts, latency, fps, captures):
    def chip(name, color_hex):
        n = counts.get(name, 0)
        return (f"<div style='flex:1;background:#1c1c1c;border-left:5px solid {color_hex};"
                f"border-radius:8px;padding:10px 12px;margin:4px'>"
                f"<div style='color:#aaa;font-size:12px'>{name}</div>"
                f"<div style='color:#fff;font-size:26px;font-weight:700'>{n}</div></div>")

    chips = (chip("fallen", "#f03c3c") + chip("no_helmet", "#ffaa00") + chip("ok", "#56c85a"))

    alerts = [c for c in cc.ALERT_CLASSES if counts.get(c, 0) > 0]
    if alerts:
        banner = (f"<div style='background:#c81e1e;color:#fff;padding:12px;border-radius:8px;"
                  f"margin:6px 4px;font-size:18px;font-weight:700;text-align:center'>"
                  f"⚠ 위험 감지: {', '.join(a + ' ' + str(counts[a]) for a in alerts)}</div>")
    else:
        banner = ("<div style='background:#1a3a1a;color:#7CFC7C;padding:12px;border-radius:8px;"
                  "margin:6px 4px;text-align:center'>✅ 정상</div>")

    meta = (f"<div style='display:flex;color:#bbb;font-size:13px;margin:6px 4px;gap:14px'>"
            f"<span>🎞 {fps:4.1f} FPS</span><span>⏱ {latency:4.0f} ms</span>"
            f"<span>📸 누적 캡처 {captures}장</span></div>")

    return (f"<div style='font-family:sans-serif'>{banner}"
            f"<div style='display:flex'>{chips}</div>{meta}</div>")


def detect(frame_rgb, conf, iou, show_label, show_conf, capture_on, capture_secs):
    if frame_rgb is None:
        return None, "<div style='color:#aaa'>카메라를 연결해 주세요.</div>"

    fps = _fps()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, counts, latency, dets = cc.detect_and_annotate(
        frame_bgr, conf=conf, iou=iou,
        show_label=show_label, show_conf=show_conf, cam_name="WEBCAM", fps=fps,
    )

    if capture_on:
        _tracker.capture_seconds = float(capture_secs)
        _tracker.update(counts, frame_bgr, annotated_bgr, "WEBCAM", dets)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, _status_html(counts, latency, fps, _tracker.saved_count)


HEADER = """
<div style="text-align:center;padding:6px">
  <h1 style="margin:0">🪖 CCTV 헬멧 감지 시스템</h1>
  <p style="color:#888;margin:4px">YOLOv8s · <b>round7</b> (mAP50 0.944) · fallen / no_helmet / ok</p>
</div>
"""

CSS = """
.gradio-container {max-width: 1200px !important}
#out_img {border-radius: 10px}
"""

with gr.Blocks(title="CCTV 헬멧 감지") as demo:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=3):
            webcam = gr.Image(sources=["webcam"], streaming=True, label="📹 웹캠 입력", type="numpy")
            out_img = gr.Image(label="🔎 감지 결과", type="numpy", elem_id="out_img")
        with gr.Column(scale=1):
            status = gr.HTML()
            gr.Markdown("### ⚙️ 파라미터")
            conf_s = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence")
            iou_s = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU (NMS)")
            with gr.Row():
                lbl_c = gr.Checkbox(value=True, label="클래스명")
                cnf_c = gr.Checkbox(value=True, label="신뢰도")
            gr.Markdown("### 📸 지속 이벤트 자동 캡처")
            cap_on = gr.Checkbox(value=True, label="캡처 활성 (재학습용 데이터)")
            cap_s = gr.Slider(2, 30, value=10, step=1, label="캡처 임계 지속시간(초)")
            gr.Markdown("<small style='color:#888'>fallen·no_helmet 가 N초 이상 지속되면 "
                        "원본 프레임을 <code>cctv_captures/</code> 에 저장</small>")

    webcam.stream(
        fn=detect,
        inputs=[webcam, conf_s, iou_s, lbl_c, cnf_c, cap_on, cap_s],
        outputs=[out_img, status],
        time_limit=30,
        stream_every=0.05,  # ~20 fps
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False,
        theme=gr.themes.Soft(primary_hue="red"), css=CSS,
    )
