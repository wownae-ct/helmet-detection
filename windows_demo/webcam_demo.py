"""
웹캠 실시간 2-stage 데모 (Windows)
실행:  python webcam_demo.py                 # 기본 카메라
       python webcam_demo.py --src 1         # 다른 카메라
       python webcam_demo.py --ok-thresh 0.95
q = 종료.  빨강=no_helmet(맨머리/캡 위반) · 초록=ok(헬멧) · 주황=fallen
"""
import argparse, time
import cv2
import two_stage as ts   # 같은 폴더

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='0')
    ap.add_argument('--ok-thresh', type=float, default=ts.OK_THRESH)
    a = ap.parse_args()
    ts.OK_THRESH = a.ok_thresh
    src = int(a.src) if a.src.isdigit() else a.src
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)   # 윈도우 카메라 백엔드
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: {src}")
    print(f"시작 (OK_THRESH={ts.OK_THRESH}). q=종료")
    t0, n = time.time(), 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        out = ts.draw(fr, ts.detect_two_stage(fr))
        n += 1; fps = n / (time.time() - t0 + 1e-6)
        cv2.putText(out, f"FPS {fps:.1f}  OKthr {ts.OK_THRESH}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("2-stage helmet (q=quit)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
