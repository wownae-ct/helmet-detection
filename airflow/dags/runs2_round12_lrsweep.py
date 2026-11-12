"""
helmet_detection_round12_dag — lr 스윕 (round11 균형데이터 위에서 lr만 탐색)
=========================================================================
round11(균형데이터, lr 0.003)이 끝난 뒤, '같은 데이터·같은 base'에서 lr만 바꿔
0.001 / 0.002 를 학습 → round11(0.003)과 비교해 '최적 lr'을 찾는다.
변수 = lr 하나만 (데이터·base·증강 전부 round11과 동일) → 깨끗한 lr 스윕.

각 lr: round6 best ──fine-tune→ data_sample/round11(균형)  → 검증(test) → 백업 → MLflow
run 이름: round12_lr001, round12_lr002
"""
from __future__ import annotations
import shutil
from datetime import datetime as dt
from pathlib import Path
from airflow.sdk import dag, task

PROJECT = "/mnt/ml/helmet_project"
RUNS_DIR = f"{PROJECT}/experiments/runs2"
DATA = f"{PROJECT}/data/data_sample/round11/data.yaml"   # round11과 동일 균형데이터
BASE = f"{PROJECT}/experiments/runs2/round6/weights/best.pt"
MLFLOW_URI = "http://127.0.0.1:5000"
VAL_CONF = 0.25
BACKUP_THRESHOLD = 0.85
BACKUP_DIR = f"{PROJECT}/models/best/best_models"
CLASS_NAMES = ['fallen', 'no_helmet', 'ok']
LR_SWEEP = [0.001, 0.002]   # round11=0.003 과 비교

BASE_ARGS = dict(epochs=100, batch=32, imgsz=640, lrf=0.01, warmup_epochs=1, cls=2.0,
                 patience=20, optimizer='auto', seed=0, degrees=15, scale=0.7, mosaic=1.0,
                 mixup=0.15, copy_paste=0.1, hsv_v=0.5, fliplr=0.5, flipud=0.0, close_mosaic=10)


@dag(schedule=None, catchup=False, tags=['runs2', 'round12', 'lr-sweep'],
     description='round12: 균형데이터 위 lr 스윕 (0.001/0.002 vs round11 0.003)')
def helmet_detection_round12():

    @task
    def wait_for_gpu() -> bool:
        import time, subprocess
        for _ in range(120):
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
    def sweep(_gpu):
        import mlflow
        from mlflow.tracking import MlflowClient
        from ultralytics import YOLO
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("helmet_detection")
        cl = MlflowClient()
        results = {}
        for lr in LR_SWEEP:
            name = f"round12_lr{str(lr).split('.')[1]}"   # round12_lr001, round12_lr002
            args = dict(lr0=lr, **BASE_ARGS)
            model = YOLO(BASE)
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({'base_model': BASE, 'data': DATA, 'round': name,
                                   'note': f'lr-sweep on balanced data (lr={lr})',
                                   'classes': 'fallen/no_helmet/ok', 'nc': 3, **args})

                def cb(trainer):
                    try:
                        m = {k.replace('/', '_').replace('(B)', ''): float(v)
                             for k, v in trainer.metrics.items()}
                        mlflow.log_metrics(m, step=trainer.epoch)
                    except Exception:
                        pass
                model.add_callback("on_fit_epoch_end", cb)
                model.train(data=DATA, project=RUNS_DIR, name=name, exist_ok=True,
                            device='0', workers=16, **args)
                best_pt = f"{RUNS_DIR}/{name}/weights/best.pt"
                fm = model.trainer.metrics
                val_map = float(fm.get('metrics/mAP50(B)', 0))
                mlflow.log_metrics({'final_mAP50': val_map, 'val_mAP50': val_map,
                                    'final_precision': float(fm.get('metrics/precision(B)', 0)),
                                    'final_recall': float(fm.get('metrics/recall(B)', 0))})
                # test 검증
                r = YOLO(best_pt).val(data=DATA, split='test', conf=VAL_CONF, iou=0.6, max_det=300)
                test_map = float(r.results_dict.get('metrics/mAP50(B)', 0))
                cmap = {}
                try:
                    for j, ci in enumerate(r.box.ap_class_index):
                        cmap[CLASS_NAMES[int(ci)]] = float(r.box.ap50[j])
                except Exception:
                    pass
                cl.log_metric(run.info.run_id, 'test_mAP50', test_map)
                for c, v in cmap.items():
                    cl.log_metric(run.info.run_id, f'test_mAP50_{c}', v)
                mlflow.log_artifact(best_pt, artifact_path='weights')
                # 백업
                if test_map >= BACKUP_THRESHOLD:
                    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(best_pt, f"{BACKUP_DIR}/{name}_map{test_map:.3f}_{dt.now():%Y%m%d_%H%M}.pt")
                results[name] = {'lr': lr, 'val': val_map, 'test': test_map, 'class': cmap}
                print(f"✅ {name}: val {val_map:.4f} / test {test_map:.4f}  {cmap}")
        print(f"\n{'='*56}\n  round12 lr 스윕 결과 (균형데이터)\n{'='*56}")
        print(f"  {'run':16}{'lr':>8}{'test mAP50':>12}")
        for n, d in results.items():
            print(f"  {n:16}{d['lr']:>8}{d['test']:>12.4f}")
        print(f"  (참고) round11 lr0.003 / round7 0.957 / round6 0.965")
        return results

    sweep(wait_for_gpu())


helmet_detection_round12()
