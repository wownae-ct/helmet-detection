# helmet-detection

YOLOv8 기반으로 산업현장의 **안전모 미착용·쓰러짐**을 탐지하는 모델을 여러 라운드에 걸쳐 학습·비교한 기록입니다. 데이터 준비 스크립트, Airflow 학습 DAG, MLflow 기록 유틸, 2단계 추론 데모, 라운드별 실험 결과(CSV·그래프)를 담았습니다.

> **2026-06 프로젝트의 정리본입니다.** 데이터셋, 모델 가중치(`.pt`), MLflow·Airflow DB는 포함하지 않았습니다(원래 저장소는 약 927MB였고, 이 저장소는 코드·문서·실험 증거만 담아 약 8MB입니다). 데이터에는 외부 공개 데이터(roboflow, kaggle)와 팀 라벨링 데이터가 섞여 있어 올리지 않았습니다. **그래서 이 저장소만으로 학습이나 추론을 그대로 재현할 수는 없습니다.** 전체 학습 이력과 가중치는 비공개 아카이브에 있습니다.

## 이 저장소에 있는 것 / 없는 것

| 있음 | 없음 |
|---|---|
| `src/`(데이터 준비·추론·분석·MLflow 유틸 스크립트) | 데이터셋 (약 17,500장) |
| `airflow/dags/`(round 9~12 학습 DAG 3개) | 모델 가중치 `.pt` |
| `experiments/runs2/round*/`의 `args.yaml`, `results.csv`, 학습 곡선·혼동행렬 이미지 | MLflow DB(`mlflow.db`), Airflow DB·로그 |
| `notebooks/`(round1~4 학습 노트북, 출력 비움) | Azure 연동 자격 정보(`.env`는 제외, `.env-example`만 포함) |
| `docs/`(실험 과정 문서, 라운드 비교 차트) | 팀원 공동 라벨링 원본 |

## 클래스와 최종 모델

- **3클래스**: `0=fallen`(쓰러짐) · `1=no_helmet`(미착용: 맨얼굴·캡·모자) · `2=ok`(안전모 착용)
- **최종 배포 모델은 round7입니다.** 다만 지표상 round6이 동급 이상입니다. 자세한 비교는 [`docs/round_by_round_detail.md`](docs/round_by_round_detail.md)에 있습니다.
- 문서: [`docs/mlops_journey.md`](docs/mlops_journey.md)(과정 요약), [`docs/round_by_round_detail.md`](docs/round_by_round_detail.md)(라운드별 상세), [`docs/presentation_qa.md`](docs/presentation_qa.md)(발표 Q&A)

## 실험 결과 요약

공통 hold-out test 셋에서 `best.pt`를 재추론해 통일 측정한 test mAP50입니다(conf 0.25 기준).

| round | test mAP50 | 메모 |
|---|---|---|
| 4 | 0.863 | baseline |
| 5 | 0.866 | |
| 6 | **0.965** | 증강으로 과적합 해소. test·안전 지표 1위 |
| 7 | 0.957 | 전체 데이터로 학습한 최종 배포 모델 |

- round1~3은 연습(2클래스 검증 → 3클래스 첫 시도 실패)입니다.
- round8(scratch), round9·10(fallen 강화), round11(균형 데이터), round12(lr 스윕), 앙상블은 **어느 것도 round6을 넘지 못했습니다.**
- **측정 기준 주의**: 지표는 conf 0.25(재현율 우선, 안전 도메인) 기준입니다. conf 0.5로 채점한 옛 기록은 값이 더 낮게 나옵니다(같은 모델, 채점 기준 차이). 라운드 간 비교는 같은 hold-out test 셋으로 했습니다.
- 데이터 규모: 전체 17,541장(train 12,855 / valid 3,530 / test 1,156).

## 파이프라인

```
데이터 수집 → 전처리/병합 → 학습(YOLOv8) → 검증/백업 → 추론/배포
roboflow·Azure·kaggle   src/data/   Airflow DAG + MLflow   test셋, conf 0.25   2-stage·Gradio·웹캠
            ▲                              │
            └──── 지표 비교 → 다음 라운드 설계 ◄──┘
```

- **자동화(Airflow)**: 데이터 준비 → GPU 대기 → 학습 → 검증(클래스별 mAP) → best 백업 → MLflow 기록.
- **사람이 판단한 것**: 데이터 균형 설계, 증강 전략, 최종 모델 채택은 실험 결과를 보고 수동으로 결정했습니다.

## 데이터 준비 (`src/data/`)

| 스크립트 | 역할 |
|---|---|
| `01_remap_roboflow.py` | roboflow `[head, helmet]` → `[no_helmet, ok]` 클래스 리매핑 |
| `02_azure_to_yolo.py` | Azure(팀 라벨링 모자) COCO → YOLO 포맷 변환 |
| `03_merge_3class.py` | 전 소스를 3클래스로 병합·리매핑 |
| `04_balance.py` | 클래스 밸런싱 |
| `05_verify.py` | 라벨·분포 검증 |
| `06_sample_round5_6.py` | 균형 샘플 추출(round5·6) |
| `07_sample_round11.py` | 균형 전체 데이터 샘플(round11) |
| `08_analyze_dist.py` | 클래스 분포 분석 |

## 학습 DAG (`airflow/dags/`)

각 DAG는 같은 흐름을 따릅니다: `prepare_dataset → wait_for_gpu → train → validate(클래스별 mAP) → backup(임계값 통과 시) → MLflow 기록`.

| 파일 | 실험 의도 |
|---|---|
| `runs2_round9_10.py` | round9→10 연속 학습. round9는 fallen 강화(회전·상하반전), round10은 round9의 클래스별 mAP를 보고 파라미터를 동적으로 조정 |
| `runs2_round11.py` | round6의 데이터 균형을 전체 최신 데이터에 적용(데이터 균형만 다른 A/B) |
| `runs2_round12_lrsweep.py` | round11 데이터 위에서 lr만 0.001/0.002로 탐색(round11은 0.003) |

핵심 설정(round9·10 기준): `epochs=100, batch=40, imgsz=640, cls=2.0`.

## 실험 기록 (MLflow)

- 모든 학습을 MLflow `experiment='helmet_detection'`으로 기록했습니다. 파라미터, 에포크별 지표(mAP50, precision, recall, loss), 최종 지표(클래스별 mAP 포함), 아티팩트를 남깁니다.
- `src/mlflow_utils/backfill_metrics.py`: 과거 라운드의 빈 지표를 `results.csv`와 재추론으로 채웁니다.
- `src/mlflow_utils/upload_artifacts.py`: 혼동행렬·PR 곡선·`results.png`를 MLflow에 올립니다.
- MLflow 서버를 직접 띄우려면: `mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns --host 127.0.0.1 --port 5000`

## 추론·데모 (`src/inference/`, `src/cctv/`, `windows_demo/`)

| 모듈 | 설명 |
|---|---|
| `src/inference/two_stage.py` | 2단계 엔진: person 검출 → 머리 영역 crop → helmet 분류. fallen은 round7 모델로 병행 검출. `OK_THRESH=0.90`으로 애매하면 위반으로 처리(안전 편향) |
| `src/inference/webcam_two_stage.py` | 2단계 웹캠 실시간 |
| `src/inference/gradio_app.py` | 단일 모델 Gradio 데모 |
| `src/inference/make_cls_crops.py` | 분류기 학습용 머리 crop 생성 |
| `src/cctv/` | CCTV 실시간 관제(공통 모듈, Gradio 데모, 헤드리스 OpenCV 러너) |
| `windows_demo/` | Windows 추론 전용 데모(가중치 제외) |

모델 비교: `src/analysis/`의 `compare_eval.py`(round6 vs round7 P/R·혼동행렬), `compare_visual.py`, `ensemble_compare.py`(단일 vs WBF 앙상블).

## 폴더 구조

```
src/                데이터 준비, 추론, 분석, MLflow 유틸
airflow/dags/       학습 DAG (round 9~12)
experiments/
  runs2/round*/     라운드별 args.yaml, results.csv, 학습 곡선·혼동행렬 이미지
  results/, plots/  라운드 비교 그래프
notebooks/          round1~4 학습 노트북(출력 비움), OpenCV 실습
docs/               과정 문서, 라운드 비교 차트
windows_demo/       Windows 추론 데모 코드
```

## 한계

- **재현 불가**: 데이터셋, 가중치, DB가 없어서 학습·추론을 그대로 실행할 수 없습니다. 코드와 실험 결과의 구조를 보여주는 저장소입니다.
- **절대경로**: DAG와 일부 스크립트가 원본 환경의 경로(`/mnt/ml/helmet_project/...`)를 가정합니다. 다른 환경에서 쓰려면 경로를 바꿔야 합니다.
- **지표**: 모든 수치는 같은 hold-out test 셋에서 conf 0.25로 측정했고, 데이터 규모(test 1,156장)가 작아 소수점 셋째 자리 차이에는 의미를 두지 않았습니다.
- **팀 프로젝트의 일부**: 이 저장소는 모델 학습·실험 부분입니다. 팀의 서비스(대시보드·알림 등)는 별도 팀 레포에 있습니다.
