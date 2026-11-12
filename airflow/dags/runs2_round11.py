"""
helmet_detection_round11_dag — B 실험 (균형 + 전체데이터)
==========================================================
가설: round7(전체데이터, ok 68% 편중)이 test에서 round6보다 살짝 낮았다.
      → round6의 '균형'을 전체 최신데이터에 적용하면 둘의 장점을 합쳐 더 나을 수 있다.

round11 = round6 best.pt  ──fine-tune→  data_sample/round11 (균형 전체데이터)
  · base/lr/epochs/aug 는 round7과 '동일' → 차이는 오직 '데이터 균형' (깨끗한 A/B).
  · valid/test 는 data_final_backup 동일 셋 → round6/round7과 직접 비교 가능.

[자동화] 트리거 → GPU게이트 → 학습 → 검증(test, 클래스별 mAP) → 백업 → MLflow 기록
[수동]   데이터 균형 설계, 최종 채택 판단은 사람.
"""
from __future__ import annotations
import os, shutil, subprocess
from datetime import datetime as dt
from pathlib import Path
from airflow.sdk import dag, task

PROJECT = "/mnt/ml/helmet_project"
RUNS_DIR = f"{PROJECT}/experiments/runs2"
DATA = f"{PROJECT}/data/data_sample/round11/data.yaml"
BASE = f"{PROJECT}/experiments/runs2/round6/weights/best.pt"   # round7과 동일 출발점
RUN_NAME = "round11"
MLFLOW_URI = "http://127.0.0.1:5000"
VAL_CONF = 0.25
BACKUP_THRESHOLD = 0.85
BACKUP_DIR = f"{PROJECT}/models/best/best_models"
CLASS_NAMES = ['fallen', 'no_helmet', 'ok']

# round7과 동일한 학습 레시피 (데이터 균형만 다름)
TRAIN_ARGS = dict(epochs=100, batch=32, imgsz=640, lr0=0.003, lrf=0.01,
                  warmup_epochs=1, cls=2.0, patience=20, optimizer='auto', seed=0,
                  degrees=15, scale=0.7, mosaic=1.0, mixup=0.15, copy_paste=0.1,
                  hsv_v=0.5, fliplr=0.5, flipud=0.0, close_mosaic=10)


@dag(schedule=None, catchup=False, tags=['runs2', 'round11', 'B-balanced'],
     description='round11: round6 best → 균형 전체데이터 fine-tune (B 실험)')
def helmet_detection_round11():

    @task
    def wait_for_gpu() -> bool:
        import time
        for _ in range(60):
            try:
                used = int(subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0])
                if used < 3000:
                    return True
            except Exception:
                return True
            time.sleep(30)
        return True

    @task
    def train(_gpu) -> dict:
        import mlflow
        from ultralytics import YOLO
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("helmet_detection")
        model = YOLO(BASE)
        with mlflow.start_run(run_name=RUN_NAME) as run:
            mlflow.log_params({'base_model': BASE, 'data': DATA, 'round': 11,
                               'classes': 'fallen/no_helmet/ok', 'nc': 3,
                               'note': 'B-balanced(round6 base + full-balanced data)', **TRAIN_ARGS})

            def on_epoch_end(trainer):
                try:
                    m = {k.replace('/', '_').replace('(B)', ''): float(v)
                         for k, v in trainer.metrics.items()}
                    mlflow.log_metrics(m, step=trainer.epoch)
                except Exception:
                    pass
            model.add_callback("on_fit_epoch_end", on_epoch_end)

            model.train(data=DATA, project=RUNS_DIR, name=RUN_NAME, exist_ok=True,
                        device='0', workers=16, **TRAIN_ARGS)
            best_pt = f"{RUNS_DIR}/{RUN_NAME}/weights/best.pt"
            final = model.trainer.metrics
            map50 = float(final.get('metrics/mAP50(B)', 0))
            mlflow.log_metrics({'final_mAP50': map50,
                                'final_precision': float(final.get('metrics/precision(B)', 0)),
                                'final_recall': float(final.get('metrics/recall(B)', 0)),
                                'val_mAP50': map50})
            mlflow.log_artifact(best_pt, artifact_path='weights')
            run_id = run.info.run_id
        print(f"✅ {RUN_NAME} 완료  val_mAP50={map50:.4f}")
        return {'best_pt': best_pt, 'run_id': run_id}

    @task
    def validate(tr: dict) -> dict:
        import mlflow
        from mlflow.tracking import MlflowClient
        from ultralytics import YOLO
        r = YOLO(tr['best_pt']).val(data=DATA, split='test', conf=VAL_CONF, iou=0.6, max_det=300)
        m = r.results_dict
        map50 = float(m.get('metrics/mAP50(B)', 0))
        class_maps = {}
        try:
            for j, ci in enumerate(r.box.ap_class_index):
                class_maps[CLASS_NAMES[int(ci)]] = float(r.box.ap50[j])
        except Exception:
            pass
        mlflow.set_tracking_uri(MLFLOW_URI)
        cl = MlflowClient()
        cl.log_metric(tr['run_id'], 'test_mAP50', map50)
        for c, v in class_maps.items():
            cl.log_metric(tr['run_id'], f'test_mAP50_{c}', v)
        print(f"\n📊 [{RUN_NAME}] test mAP50={map50:.4f} (conf {VAL_CONF})  클래스별={class_maps}")
        print(f"   비교: round6 test 0.965 / round7 test 0.957 (conf 0.25 기준)")
        return {'map50': map50, 'best_pt': tr['best_pt'], 'class_maps': class_maps}

    @task
    def backup(vr: dict) -> dict:
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        map50 = vr['map50']
        if map50 >= BACKUP_THRESHOLD:
            dst = f"{BACKUP_DIR}/{RUN_NAME}_map{map50:.3f}_{dt.now().strftime('%Y%m%d_%H%M')}.pt"
            shutil.copy2(vr['best_pt'], dst)
            print(f"✅ 백업: {dst}")
        else:
            print(f"⚠️ mAP50 {map50:.4f} < {BACKUP_THRESHOLD} — 백업 스킵")
        return vr

    @task
    def summary(vr: dict):
        print(f"\n{'='*56}\n  round11 (B: 균형+전체데이터) 결과\n{'='*56}")
        print(f"  test mAP50 = {vr['map50']:.4f}")
        for c in CLASS_NAMES:
            print(f"    {c:12s}: {vr['class_maps'].get(c, 'N/A')}")
        print(f"\n  목표: round7(0.957)·round6(0.965) 대비 향상 여부 확인")
        print(f"  → MLflow UI에서 round6/round7/round11 나란히 비교")

    g = wait_for_gpu()
    t = train(g)
    v = validate(t)
    b = backup(v)
    summary(b)


helmet_detection_round11()
