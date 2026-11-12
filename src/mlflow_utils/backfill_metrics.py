"""
MLflow 백필 — runs2 round1~10 의 빈 지표/파라미터 채우기
  ① round 파라미터(round4~10 누락) ← run 이름
  ② final_mAP50/precision/recall/mAP50_95(round1~5 누락) ← results.csv 최종 epoch
  ③ test_mAP50(+클래스별)(round1~10 누락) ← best.pt 로 test셋 추론
공정 비교: 3클래스 라운드(3~10)는 공통 test셋(data_final_backup), 2클래스(1·2)는 data_remapped.
"""
import os, warnings
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
RUNS = "/mnt/ml/helmet_project/experiments/runs2"
TEST_3 = "/mnt/ml/helmet_project/data/data_final_backup/data.yaml"  # 공통 3클래스 hold-out
TEST_2 = "/mnt/ml/helmet_project/data/data_remapped/data.yaml"
NAMES3 = {0: 'fallen', 1: 'no_helmet', 2: 'ok'}

exp = client.get_experiment_by_name("helmet_detection")

def get_run(name):
    rs = client.search_runs([exp.experiment_id],
                            filter_string=f"tags.mlflow.runName = '{name}'", max_results=1)
    return rs[0] if rs else None

for i in range(1, 11):
    nm = f"round{i}"
    run = get_run(nm)
    if not run:
        print(f"{nm}: run 없음 — 스킵"); continue
    rid = run.info.run_id
    params, metrics = run.data.params, run.data.metrics
    did = []

    # ① round 파라미터
    if 'round' not in params:
        client.log_param(rid, 'round', i); did.append('round')

    # ② final_* (results.csv)
    csv = f"{RUNS}/{nm}/results.csv"
    if 'final_mAP50' not in metrics and os.path.exists(csv):
        df = pd.read_csv(csv); df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        for key, col in [('final_mAP50', 'metrics/mAP50(B)'),
                         ('final_mAP50_95', 'metrics/mAP50-95(B)'),
                         ('final_precision', 'metrics/precision(B)'),
                         ('final_recall', 'metrics/recall(B)')]:
            if col in df.columns:
                client.log_metric(rid, key, float(last[col]))
        did.append('final_*')

    # ③ test_mAP50 (+클래스별) — best.pt 추론
    best = f"{RUNS}/{nm}/weights/best.pt"
    if 'test_mAP50' not in metrics and os.path.exists(best):
        data = TEST_2 if i in (1, 2) else TEST_3
        try:
            res = YOLO(best).val(data=data, split='test', conf=0.25, iou=0.6, verbose=False)
            client.log_metric(rid, 'test_mAP50', float(res.results_dict.get('metrics/mAP50(B)', 0)))
            client.log_metric(rid, 'test_mAP50_95', float(res.results_dict.get('metrics/mAP50-95(B)', 0)))
            if getattr(res, 'ap_class_index', None) is not None and hasattr(res, 'box'):
                for j, ci in enumerate(res.ap_class_index):
                    if j < len(res.box.ap50):
                        cname = NAMES3.get(int(ci), f'cls{ci}') if i not in (1, 2) else f'cls{ci}'
                        client.log_metric(rid, f'test_mAP50_{cname}', float(res.box.ap50[j]))
            did.append('test_*')
        except Exception as e:
            print(f"{nm}: test 실패 — {e}")

    print(f"{nm}: 채움 {did if did else '(이미 채워짐)'}")

print("\n✅ 백필 완료 — MLflow UI 새로고침")
