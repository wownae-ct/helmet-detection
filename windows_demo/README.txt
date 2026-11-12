============================================================
 2-stage 헬멧 감지 — Windows 웹캠 데모
============================================================

[폴더 구성]
  webcam_demo.py     ← 실행 파일 (웹캠)
  two_stage.py       ← 추론 엔진 (webcam_demo 가 자동 호출)
  models/
    yolov8s.pt       ← COCO person 검출
    helmet_cls.pt    ← 머리크롭 헬멧 분류기
    fallen_round7.pt ← fallen 검출
  → 폴더째로 복사하면 어디서든 동작 (경로는 모두 상대경로).

[1] 의존성 설치 (해당 venv 에서 1회)
    pip install ultralytics opencv-python

[2] 실행 (이 폴더 안에서)
    cd C:\helmet_demo
    python webcam_demo.py

    옵션:
      python webcam_demo.py --src 1        (카메라 번호 변경)
      python webcam_demo.py --ok-thresh 0.95   (헬멧 판정 더 엄격 → 위반 더 많이 잡음)
      python webcam_demo.py --ok-thresh 0.85   (오경보 줄임)

[색]  빨강 = no_helmet(맨머리/캡, 위반)
      초록 = ok(헬멧)
      주황 = fallen(쓰러짐)
   q 키 = 종료

[이미지/영상으로 테스트]
    python two_stage.py --image 사진.jpg --out 결과.jpg
    python two_stage.py --video 영상.mp4 --out 결과.jpg

[주의] 기존 test.py 는 WSL 절대경로를 임포트해서 윈도우에서 실패합니다.
       test.py 대신 webcam_demo.py 를 쓰세요. (또는 test.py 를
       'import two_stage as ts' 로 고치고 이 폴더에서 실행)
