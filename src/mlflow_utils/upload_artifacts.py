"""
DAG로 학습한 round4~12 의 YOLO 학습 그림(confusion_matrix.png 등)을 MLflow Artifacts에 업로드.
(round1~3 노트북 run은 이미 올라가 있음. DAG run은 weights만 올려서 그림 누락 → 보완)
발표 캡처용: confusion_matrix.png(클래스별 혼동), BoxPR_curve.png(P-R곡선), results.png(loss·mAP 곡선)
"""
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
cl = MlflowClient()
exp = cl.get_experiment_by_name("helmet_detection")
RUNS = Path("/mnt/ml/helmet_project/experiments/runs2")
FILES = ['confusion_matrix.png', 'confusion_matrix_normalized.png',
         'BoxPR_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 'BoxF1_curve.png',
         'results.png', 'results.csv', 'args.yaml',
         'val_batch0_pred.jpg', 'val_batch0_labels.jpg']

# round12 는 run 이름이 round12_lr001/002
names = sys.argv[1:] or [f"round{i}" for i in range(4, 12)] + ['round12_lr001', 'round12_lr002']


def main_run(nm):
    rs = cl.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName='{nm}'", max_results=10)
    return max(rs, key=lambda r: len(r.data.metrics)).info.run_id if rs else None


for nm in names:
    d = RUNS / nm
    if not d.exists():
        print(f"{nm}: 학습 폴더 없음 — 스킵")
        continue
    rid = main_run(nm)
    if not rid:
        print(f"{nm}: MLflow run 없음 — 스킵")
        continue
    existing = {a.path for a in cl.list_artifacts(rid)}
    up = []
    for f in FILES:
        p = d / f
        if p.exists() and f not in existing:
            cl.log_artifact(rid, str(p))
            up.append(f)
    print(f"{nm}: 업로드 {up if up else '(이미 있음/파일없음)'}")

print("\n✅ 완료 — MLflow UI에서 각 run → Artifacts 탭 확인")
