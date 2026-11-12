"""
헬멧 감지 Airflow DAG — round9 → round10 자동 연속 학습

round9 : round7/best.pt → data_final_backup 전체
         fallen 클래스 강화 (degrees↑, flipud 추가)
round10: round9/best.pt → data_final_backup 전체
         round9 클래스별 mAP를 보고 파라미터 동적 조정

공통:
  batch=40, workers=16, cache='ram'
  전체 데이터 17,541장 (train 12,855 / valid 3,530 / test 1,156)

사용법:
  cp helmet_detection_dag_r9r10.py ~/airflow/dags/
  Airflow UI에서 helmet_detection_r9r10 트리거
"""

from datetime import datetime, timedelta
from airflow.sdk import dag, task

# ── 공통 설정 ─────────────────────────────────────────────
PROJECT_DIR = '/mnt/ml/helmet_project'
RUNS_DIR    = f'{PROJECT_DIR}/experiments/runs2'
MLFLOW_URI  = 'http://127.0.0.1:5000'
EPOCHS      = 100
BATCH       = 40
WORKERS     = 16
DATA        = f'{PROJECT_DIR}/data/data_final_backup/data.yaml'

# ── round9 고정 aug 파라미터 (fallen 강화) ─────────────────
AUG_ROUND9 = {
    'hsv_h'     : 0.03,
    'hsv_s'     : 0.8,
    'hsv_v'     : 0.5,
    'degrees'   : 30,    # 기존 15 → 30 (fallen=쓰러진 사람, 회전 다양성)
    'scale'     : 0.7,
    'fliplr'    : 0.5,
    'flipud'    : 0.3,   # 추가 (기존 0.0)
    'mosaic'    : 1.0,
    'mixup'     : 0.15,
    'copy_paste': 0.1,
}

# ── round10 기본 aug (동적 조정의 베이스) ──────────────────
AUG_ROUND10_BASE = {
    'hsv_h'     : 0.03,
    'hsv_s'     : 0.8,
    'hsv_v'     : 0.5,
    'scale'     : 0.7,
    'fliplr'    : 0.5,
    'mosaic'    : 1.0,
    'mixup'     : 0.15,
    'copy_paste': 0.1,
    # degrees, flipud는 동적으로 결정
}

default_args = {
    'owner'      : 'helmet_team',
    'retries'    : 1,
    'retry_delay': timedelta(minutes=5),
}

CLASS_NAMES = ['fallen', 'no_helmet', 'ok']


@dag(
    dag_id       = 'helmet_detection_r9r10',
    default_args = default_args,
    description  = 'round9(fallen강화) → round10(동적파라미터) 전체 데이터셋',
    schedule     = None,
    start_date   = datetime(2026, 1, 1),
    catchup      = False,
    tags         = ['helmet', 'yolo', 'mlflow', 'round9', 'round10'],
)
def helmet_pipeline_r9r10():

    # ── 공통 유틸: 에포크별 MLflow 로깅 ───────────────────
    def _log_epoch_metrics(trainer, mlflow_module):
        epoch    = trainer.epoch
        metrics  = trainer.metrics
        loss     = trainer.loss_items
        log_dict = {}
        if metrics:
            log_dict.update({
                'val_mAP50'    : float(metrics.get('metrics/mAP50(B)',     0)),
                'val_mAP50_95' : float(metrics.get('metrics/mAP50-95(B)', 0)),
                'val_precision': float(metrics.get('metrics/precision(B)', 0)),
                'val_recall'   : float(metrics.get('metrics/recall(B)',    0)),
                'val_box_loss' : float(metrics.get('val/box_loss',         0)),
                'val_cls_loss' : float(metrics.get('val/cls_loss',         0)),
                'val_dfl_loss' : float(metrics.get('val/dfl_loss',         0)),
            })
        if loss is not None and len(loss) >= 3:
            log_dict.update({
                'train_box_loss': float(loss[0]),
                'train_cls_loss': float(loss[1]),
                'train_dfl_loss': float(loss[2]),
            })
        if log_dict:
            mlflow_module.log_metrics(log_dict, step=epoch)

    # ── 공통 유틸: 학습 ────────────────────────────────────
    def _train(run_name, base_model, lr0, warmup_epochs, dist, aug_params, patience=25) -> dict:
        import mlflow
        import functools
        from ultralytics import YOLO
        from ultralytics import settings as yolo_settings
        yolo_settings.update({'mlflow': False})

        print(f"\n{'='*56}")
        print(f"  {run_name} 학습 시작")
        print(f"  베이스: {base_model}")
        print(f"  데이터: {DATA}")
        print(f"  batch={BATCH}, workers={WORKERS}, cache='ram'")
        print(f"  lr0={lr0}, warmup={warmup_epochs}, patience={patience}")
        print(f"  aug: degrees={aug_params.get('degrees')}, "
              f"flipud={aug_params.get('flipud', 0.0)}")
        print(f"{'='*56}")

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment('helmet_detection')

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                'base_model'    : base_model,
                'data'          : DATA,
                'epochs'        : EPOCHS,
                'imgsz'         : 640,
                'batch'         : BATCH,
                'workers'       : WORKERS,
                'cache'         : 'ram',
                'lr0'           : lr0,
                'warmup_epochs' : warmup_epochs,
                'patience'      : patience,
                'cls_weight'    : 2.0,
                'nc'            : 3,
                'classes'       : 'fallen/no_helmet/ok',
                'fallen_cnt'    : dist.get(0, 0),
                'no_helmet_cnt' : dist.get(1, 0),
                'ok_cnt'        : dist.get(2, 0),
                **{f'aug_{k}': v for k, v in aug_params.items()},
            })

            model = YOLO(base_model)
            model.add_callback(
                'on_fit_epoch_end',
                functools.partial(_log_epoch_metrics, mlflow_module=mlflow)
            )

            results = model.train(
                data          = DATA,
                epochs        = EPOCHS,
                imgsz         = 640,
                batch         = BATCH,
                workers       = WORKERS,
                cache         = 'ram',
                device        = 0,
                project       = RUNS_DIR,
                name          = run_name,
                exist_ok      = True,
                cls           = 2.0,
                lr0           = lr0,
                warmup_epochs = warmup_epochs,
                patience      = patience,
                **aug_params,
            )

            final   = results.results_dict
            best_pt = f'{RUNS_DIR}/{run_name}/weights/best.pt'
            map50   = float(final.get('metrics/mAP50(B)', 0))

            mlflow.log_metrics({
                'final_mAP50'    : map50,
                'final_mAP50_95' : float(final.get('metrics/mAP50-95(B)', 0)),
                'final_precision': float(final.get('metrics/precision(B)', 0)),
                'final_recall'   : float(final.get('metrics/recall(B)',    0)),
            })
            mlflow.log_artifact(best_pt, artifact_path='weights')

            print(f"\n✅ {run_name} 완료  mAP50: {map50:.4f}")
            print(f"   Run ID: {run.info.run_id}")

        return {'run_name': run_name, 'best_pt': best_pt, 'mAP50': map50}

    # ── 공통 유틸: 검증 (클래스별 mAP 포함) ───────────────
    def _validate(train_result: dict) -> dict:
        from ultralytics import YOLO

        run_name = train_result['run_name']
        best_pt  = train_result['best_pt']

        model   = YOLO(best_pt)
        results = model.val(data=DATA, split='test', conf=0.5)
        metrics = results.results_dict
        map50   = float(metrics.get('metrics/mAP50(B)', 0))

        # 클래스별 mAP50 추출
        class_maps = {}
        try:
            if (hasattr(results, 'ap_class_index')
                    and results.ap_class_index is not None
                    and hasattr(results, 'box')
                    and hasattr(results.box, 'ap50')):
                for i, cls_idx in enumerate(results.ap_class_index):
                    if i < len(results.box.ap50):
                        class_maps[CLASS_NAMES[cls_idx]] = float(results.box.ap50[i])
        except Exception as e:
            print(f"⚠️  클래스별 mAP 추출 실패: {e}")

        print(f"\n📊 테스트 [{run_name}]")
        print(f"   mAP50    : {map50:.4f}")
        print(f"   Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"   Recall   : {metrics.get('metrics/recall(B)', 0):.4f}")
        print(f"   ── 클래스별 mAP50 ──")
        for cls in CLASS_NAMES:
            v = class_maps.get(cls, -1)
            mark = '⚠️ ' if cls == 'fallen' and v < 0.90 else '✅ '
            print(f"   {mark}{cls:12s}: {v:.4f}" if v >= 0 else f"   ❓ {cls:12s}: N/A")

        return {
            'map50'     : map50,
            'best_pt'   : best_pt,
            'run_name'  : run_name,
            'class_maps': class_maps,
        }

    # ── 공통 유틸: 백업 ────────────────────────────────────
    def _backup(val_result: dict, threshold: float) -> dict:
        import shutil
        from pathlib import Path
        from datetime import datetime as dt

        map50    = val_result['map50']
        best_pt  = val_result['best_pt']
        run_name = val_result['run_name']

        backup_dir = Path(f'{PROJECT_DIR}/models/best')
        backup_dir.mkdir(exist_ok=True)

        if map50 >= threshold:
            ts  = dt.now().strftime('%Y%m%d_%H%M')
            dst = backup_dir / f'{run_name}_map{map50:.3f}_{ts}.pt'
            shutil.copy2(best_pt, dst)
            print(f"✅ 백업 완료: {dst}")
        else:
            print(f"⚠️  mAP50 {map50:.4f} < {threshold} — 백업 스킵")

        # class_maps도 함께 전달 (round10 동적 조정에 필요)
        return {
            'map50'     : str(map50),
            'class_maps': val_result.get('class_maps', {}),
            'run_name'  : run_name,
        }

    # ── Task 정의 ──────────────────────────────────────────

    @task
    def prepare_dataset():
        """data_final_backup data.yaml + 분포 확인"""
        from pathlib import Path
        from collections import defaultdict

        p = Path(DATA)
        if not p.exists():
            raise FileNotFoundError(
                f"❌ {DATA} 없음. 먼저 실행하세요:\n"
                f"cat > {DATA} << 'EOF'\n"
                f"path: {PROJECT_DIR}/data/data_final_backup\n"
                f"train: train/images\nval:   valid/images\ntest:  test/images\n"
                f"nc: 3\nnames: ['fallen', 'no_helmet', 'ok']\nEOF"
            )

        dist = defaultdict(int)
        imgs = {}
        for split in ['train', 'valid', 'test']:
            lbl_dir = p.parent / split / 'labels'
            img_dir = p.parent / split / 'images'
            imgs[split] = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
            if lbl_dir.exists():
                for lbl in lbl_dir.glob('*.txt'):
                    with open(lbl) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                dist[int(line.split()[0])] += 1

        print(f"✅ {DATA}")
        for s in ['train', 'valid', 'test']:
            print(f"  {s}: {imgs.get(s, 0):,}장")
        print(f"  fallen    : {dist.get(0, 0):,}")
        print(f"  no_helmet : {dist.get(1, 0):,}")
        print(f"  ok        : {dist.get(2, 0):,}")
        return dict(dist)

    # ── round9 ─────────────────────────────────────────────
    @task
    def train_round9(dist: dict) -> dict:
        """round7/best.pt → fallen 강화 fine-tune
        
        변경점 vs round7:
          degrees  15 → 30  (쓰러진 자세 회전 다양성)
          flipud    0 → 0.3 (상하반전으로 fallen 패턴 다양화)
          lr0   0.003 → 0.001 (더 안정적인 fine-tune)
          patience  20 → 30  (fallen 학습에 시간 여유)
        """
        return _train(
            run_name      = 'round9',
            base_model    = f'{RUNS_DIR}/round7/weights/best.pt',
            lr0           = 0.001,
            warmup_epochs = 1,
            patience      = 30,
            dist          = dist,
            aug_params    = AUG_ROUND9,
        )

    @task
    def validate_round9(train_result: dict) -> dict:
        return _validate(train_result)

    @task
    def backup_round9(val_result: dict) -> dict:
        return _backup(val_result, threshold=0.85)

    # ── round10 ────────────────────────────────────────────
    @task
    def train_round10(dist: dict, r9_backup: dict) -> dict:
        """round9/best.pt → round9 결과 보고 동적 파라미터 결정
        
        fallen mAP50 기준:
          < 0.88  → 강화 모드  (degrees=35, flipud=0.4, lr0=0.0005)
          < 0.93  → 균형 모드  (degrees=20, flipud=0.2, lr0=0.001)
          >= 0.93 → 유지 모드  (degrees=15, flipud=0.0, lr0=0.002)
        """
        class_maps = r9_backup.get('class_maps', {})
        fallen_map = class_maps.get('fallen', 0.0)

        print(f"\n{'='*56}")
        print(f"  round10 파라미터 결정")
        print(f"  round9 클래스별 mAP50:")
        for cls in CLASS_NAMES:
            print(f"    {cls:12s}: {class_maps.get(cls, 'N/A')}")
        print(f"{'='*56}")

        # ── 동적 파라미터 결정 ──────────────────────────────
        if fallen_map < 0.88:
            mode     = '🔴 강화 모드 (fallen 아직 부족)'
            lr0      = 0.0005
            degrees  = 35
            flipud   = 0.4
            patience = 35

        elif fallen_map < 0.93:
            mode     = '🟡 균형 모드 (fallen 개선 중)'
            lr0      = 0.001
            degrees  = 20
            flipud   = 0.2
            patience = 25

        else:
            mode     = '🟢 유지 모드 (fallen 충분)'
            lr0      = 0.002
            degrees  = 15
            flipud   = 0.0
            patience = 20

        aug_r10 = {
            **AUG_ROUND10_BASE,
            'degrees': degrees,
            'flipud' : flipud,
        }

        print(f"\n  결정된 모드: {mode}")
        print(f"  lr0={lr0}, degrees={degrees}, flipud={flipud}, patience={patience}")

        return _train(
            run_name      = 'round10',
            base_model    = f'{RUNS_DIR}/round9/weights/best.pt',
            lr0           = lr0,
            warmup_epochs = 1,
            patience      = patience,
            dist          = dist,
            aug_params    = aug_r10,
        )

    @task
    def validate_round10(train_result: dict) -> dict:
        return _validate(train_result)

    @task
    def backup_round10(val_result: dict) -> dict:
        return _backup(val_result, threshold=0.85)

    # ── 최종 비교 ──────────────────────────────────────────
    @task
    def summary(r9_backup: dict, r10_backup: dict):
        r9_map  = float(r9_backup.get('map50',  0))
        r10_map = float(r10_backup.get('map50', 0))

        r9_cls  = r9_backup.get('class_maps',  {})
        r10_cls = r10_backup.get('class_maps', {})

        print(f"\n{'='*56}")
        print(f"  전체 라운드 비교 (test mAP50)")
        print(f"{'='*56}")
        print(f"  round4   0.829  샘플링, 기본")
        print(f"  round5   0.888  샘플링, 기본")
        print(f"  round6   0.906  샘플링, aug")
        print(f"  round7   0.926  전체데이터, fine-tune  ← 이전 최고")
        print(f"  round8   0.867  전체데이터, scratch    (퇴보)")
        print(f"  round9  {r9_map:.3f}  전체데이터, fallen강화 fine-tune")
        print(f"  round10 {r10_map:.3f}  전체데이터, 동적파라미터 fine-tune")
        print(f"{'='*56}")

        print(f"\n  ── 클래스별 mAP50 비교 ──")
        print(f"  {'클래스':14s} {'round7':>8} {'round9':>8} {'round10':>8}")
        ref = {'fallen': 0.896, 'no_helmet': 0.937, 'ok': 0.952}
        for cls in CLASS_NAMES:
            r7v  = ref.get(cls, 0)
            r9v  = r9_cls.get(cls,  0)
            r10v = r10_cls.get(cls, 0)
            best_mark = '← best' if r10v >= r9v and r10v >= r7v else ''
            print(f"  {cls:14s} {r7v:8.3f} {r9v:8.3f} {r10v:8.3f}  {best_mark}")

        best_round = 'round10' if r10_map >= r9_map else 'round9'
        best_map   = max(r9_map, r10_map)
        prev_best  = 0.926

        print(f"\n{'='*56}")
        if best_map > prev_best:
            print(f"  🎉 신기록! {best_round}  mAP50 {best_map:.4f}  (이전 {prev_best})")
        else:
            print(f"  → 최고: {best_round}  mAP50 {best_map:.4f}")
            print(f"  → round7 (0.926) 유지 중 — 추가 전략 검토 필요")
        print(f"{'='*56}")

    # ── DAG 흐름 ──────────────────────────────────────────
    dist = prepare_dataset()

    # round9
    r9_train  = train_round9(dist)
    r9_val    = validate_round9(r9_train)
    r9_backup = backup_round9(r9_val)

    # round10 — round9 완료 후 클래스별 결과 보고 파라미터 자동 결정
    r10_train  = train_round10(dist, r9_backup)
    r10_val    = validate_round10(r10_train)
    r10_backup = backup_round10(r10_val)

    # 최종 비교
    summary(r9_backup, r10_backup)


helmet_pipeline_r9r10()