"""
[1] round6 vs round7 '라벨된 셋' 비교 평가
================================================
카메라 프레임을 직접 라벨링한 작은 셋에서 두 모델의 클래스별 P/R/mAP +
ok↔no_helmet 오분류(안전 핵심)를 나란히 비교한다. → 사용자님 주장을 '데이터로' 검증.

[준비] 카메라 프레임 20~30장을 YOLO 형식으로 라벨링
  data/camera_eval/
    images/ *.jpg
    labels/ *.txt      (각 줄: <cls> <cx> <cy> <w> <h>, cls: 0=fallen 1=no_helmet 2=ok)
    data.yaml:
        path: /mnt/ml/helmet_project/data/camera_eval
        val: images
        nc: 3
        names: ['fallen', 'no_helmet', 'ok']
  ※ Label Studio에서 'YOLO' 형식으로 export하면 됨. (캡 쓴 얼굴/맨얼굴=no_helmet, 안전모=ok)

[실행]
  python src/analysis/compare_eval.py --data data/camera_eval/data.yaml
  python src/analysis/compare_eval.py --data data/camera_eval/data.yaml --conf 0.25
"""
import argparse, warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

NAMES = {0: 'fallen', 1: 'no_helmet', 2: 'ok'}
DEF6 = "/mnt/ml/helmet_project/experiments/runs2/round6/weights/best.pt"
DEF7 = "/mnt/ml/helmet_project/experiments/runs2/round7/weights/best.pt"


def evaluate(weights, data, split, conf, iou):
    r = YOLO(weights).val(data=data, split=split, conf=conf, iou=iou, verbose=False)
    per = {}
    for j, ci in enumerate(r.box.ap_class_index):
        per[int(ci)] = dict(p=float(r.box.p[j]), r=float(r.box.r[j]),
                            ap50=float(r.box.ap50[j]))
    return per, r.confusion_matrix.matrix  # cm[pred][true]


def cross(cm, true_c, pred_c):
    """실제=true_c 인데 예측=pred_c 로 분류한 개수 (cm[pred][true])"""
    return int(cm[pred_c][true_c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='라벨된 평가셋 data.yaml')
    ap.add_argument('--split', default='val')
    ap.add_argument('--model6', default=DEF6)
    ap.add_argument('--model7', default=DEF7)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.6)
    a = ap.parse_args()

    p6, cm6 = evaluate(a.model6, a.data, a.split, a.conf, a.iou)
    p7, cm7 = evaluate(a.model7, a.data, a.split, a.conf, a.iou)

    print(f"\n{'='*64}\n클래스별 정밀도/재현율/mAP50  (data={a.data}, conf={a.conf})\n{'='*64}")
    print(f"{'class':12}{'round6  P / R / mAP':>26}{'round7  P / R / mAP':>26}")
    for ci in (0, 1, 2):
        s6, s7 = p6.get(ci), p7.get(ci)
        f = lambda s: f"{s['p']:.3f} / {s['r']:.3f} / {s['ap50']:.3f}" if s else "  (해당 클래스 없음)"
        print(f"{NAMES[ci]:12}{f(s6):>26}{f(s7):>26}")

    print(f"\n{'='*64}\nok ↔ no_helmet 오분류 (안전 핵심)\n{'='*64}")
    print(f"{'오류 유형':38}{'round6':>9}{'round7':>9}")
    print(f"{'미착용(no_helmet)→ok 오인 [위험]':38}{cross(cm6,1,2):>9}{cross(cm7,1,2):>9}")
    print(f"{'착용(ok)→no_helmet 오인 [오경보]':38}{cross(cm6,2,1):>9}{cross(cm7,2,1):>9}")
    print(f"{'교차 오분류 합 (낮을수록 구분 잘함)':38}"
          f"{cross(cm6,1,2)+cross(cm6,2,1):>9}{cross(cm7,1,2)+cross(cm7,2,1):>9}")

    labels = ['fallen', 'no_helmet', 'ok', 'bg']
    for tag, cm in [('round6', cm6), ('round7', cm7)]:
        print(f"\n--- {tag} 혼동행렬 (행=예측, 열=실제) ---")
        print("pred\\true   " + "".join(f"{l:>11}" for l in labels))
        for i, row in enumerate(cm):
            print(f"{labels[i]:11}" + "".join(f"{int(v):>11}" for v in row))

    print("\n▶ round7이 'ok/no_helmet 구분'에서 우위면: no_helmet 정밀도·재현율↑ & 교차 오분류↓ 로 나타남.")


if __name__ == '__main__':
    main()
