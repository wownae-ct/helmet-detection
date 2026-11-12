"""
헬멧 감지 Gradio 웹캠 앱 — round4
클래스: 0=fallen  1=no_helmet  2=ok
실행: python gradio_webcam.py
"""

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import time

# ── 설정 ──────────────────────────────────────────────────
MODEL_PATH = "/mnt/ml/helmet_project/experiments/runs2/round5/weights/best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

CLASS_INFO = {
    0: {"name": "fallen",     "color": (34,  87, 255),  "emoji": "🔴"},
    1: {"name": "no_helmet",  "color": (0,  165, 255),  "emoji": "🟡"},
    2: {"name": "ok",         "color": (50, 205,  50),  "emoji": "🟢"},
}

# ── 모델 로드 (전역, 한 번만) ──────────────────────────────
print(f"모델 로드: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.overrides["verbose"] = False
print("모델 로드 완료 ✓")

# ── 추론 함수 ──────────────────────────────────────────────
def detect(frame: np.ndarray, conf: float, iou: float, show_labels: bool, show_conf: bool):
    if frame is None:
        return None, "카메라를 연결해 주세요."

    t0 = time.perf_counter()

    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    result     = results[0]
    out        = frame.copy()

    counts   = {0: 0, 1: 0, 2: 0}
    boxes_xy = result.boxes.xyxy.cpu().numpy().astype(int)
    classes  = result.boxes.cls.cpu().numpy().astype(int)
    confs    = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cls, c in zip(boxes_xy, classes, confs):
        info  = CLASS_INFO.get(cls, {"name": str(cls), "color": (200, 200, 200), "emoji": "?"})
        color = info["color"]
        counts[cls] = counts.get(cls, 0) + 1

        # 박스
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # 라벨
        if show_labels:
            label = info["name"]
            if show_conf:
                label += f" {c:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                out, label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    # 우측 상단 카운터 오버레이
    pad, line_h = 10, 26
    overlay_w   = 170
    overlay_h   = line_h * 3 + pad * 2
    oy = 10; ox = out.shape[1] - overlay_w - 10
    cv2.rectangle(out, (ox, oy), (ox + overlay_w, oy + overlay_h), (20, 20, 20), -1)
    cv2.rectangle(out, (ox, oy), (ox + overlay_w, oy + overlay_h), (80, 80, 80), 1)
    for i, (cls_id, info) in enumerate(CLASS_INFO.items()):
        txt = f"{info['emoji']}  {info['name']}: {counts.get(cls_id, 0)}"
        cv2.putText(
            out, f"{info['name']}: {counts.get(cls_id, 0)}",
            (ox + 8, oy + pad + line_h * i + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            info["color"], 1, cv2.LINE_AA,
        )

    total  = sum(counts.values())
    fallen = counts.get(0, 0)
    no_hlm = counts.get(1, 0)
    ok_cnt = counts.get(2, 0)

    status_lines = [
        f"⏱  추론: {elapsed_ms:.1f} ms",
        f"📦 탐지: {total}개",
        f"🔴 fallen   : {fallen}",
        f"🟡 no_helmet: {no_hlm}",
        f"🟢 ok       : {ok_cnt}",
    ]

    # 위험 알림
    if fallen > 0:
        status_lines.append(f"\n⚠️  쓰러진 사람 감지! ({fallen}명)")
    if no_hlm > 0:
        status_lines.append(f"⚠️  미착용자 감지! ({no_hlm}명)")

    return out, "\n".join(status_lines)


# ── Gradio UI ──────────────────────────────────────────────
DESCRIPTION = """
## 🪖 헬멧 감지 시스템 — Round 4
**모델**: YOLOv8s · **클래스**: fallen / no_helmet / ok  
**valid mAP50**: 0.938 · **test mAP50**: 0.829
"""

with gr.Blocks(title="헬멧 감지", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="웹캠 입력",
                type="numpy",
            )
            output_img = gr.Image(label="감지 결과", type="numpy")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 파라미터")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=CONF_THRESHOLD, step=0.05,
                label="Confidence threshold",
            )
            iou_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=IOU_THRESHOLD, step=0.05,
                label="IoU threshold (NMS)",
            )
            show_labels = gr.Checkbox(value=True,  label="클래스명 표시")
            show_conf   = gr.Checkbox(value=True,  label="신뢰도 표시")

            gr.Markdown("### 📊 결과")
            status_box = gr.Textbox(
                label="감지 현황",
                lines=8,
                interactive=False,
            )

    # 스트리밍 — 웹캠 프레임마다 호출
    webcam.stream(
        fn=detect,
        inputs=[webcam, conf_slider, iou_slider, show_labels, show_conf],
        outputs=[output_img, status_box],
        time_limit=30,
        stream_every=0.05,   # ~20 fps
    )

    gr.Markdown("""
---
**단축키**: 웹캠 패널 클릭 후 스페이스바로 일시정지  
**모델 경로**: `best_models/round4_map0.829_20260620_2359.pt`
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # 팀원 접근 허용
        server_port=7860,
        share=False,             # 외부 공유 필요 시 True
        inbrowser=True,
    )