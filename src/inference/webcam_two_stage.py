"""
웹캠 실시간 2-stage 데모 (직접 얼굴/캡/헬멧 테스트용)
======================================================
소속: 2-stage 추론 데모 (person 검출 → 머리 crop → 헬멧 분류 + fallen 병행).
실행:  python src/inference/webcam_two_stage.py            # 기본 웹캠(0)
       python src/inference/webcam_two_stage.py --src 1    # 다른 카메라
       python src/inference/webcam_two_stage.py --ok-thresh 0.9
q 키로 종료.  빨강=no_helmet(맨머리/캡) · 초록=ok(헬멧) · 주황=fallen.
의존: 같은 폴더의 two_stage.py(추론 엔진)를 importlib로 로드.
"""
import argparse, time, importlib.util
import cv2

# two_stage.py(추론 엔진)를 절대경로로 로드. 리팩토링: src/12_two_stage.py → src/inference/two_stage.py
spec = importlib.util.spec_from_file_location(
    "ts", "/mnt/ml/helmet_project/src/inference/two_stage.py")
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='0')
    ap.add_argument('--ok-thresh', type=float, default=ts.OK_THRESH)
    a = ap.parse_args()
    ts.OK_THRESH = a.ok_thresh
    src = int(a.src) if a.src.isdigit() else a.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: {src}")
    print(f"2-stage 웹캠 시작 (OK_THRESH={ts.OK_THRESH}). q=종료")
    t0 = time.time(); n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        dets = ts.detect_two_stage(fr)
        out = ts.draw(fr, dets)
        n += 1; fps = n / (time.time() - t0 + 1e-6)
        cv2.putText(out, f"FPS {fps:.1f}  OKthr {ts.OK_THRESH}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("2-stage helmet (q=quit)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
