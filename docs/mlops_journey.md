# 발표 상세 문서: 안전모 감지 모델 MLOps 여정 (round1 → round7)
> PPT 토대. 최종 모델 = **round7**(yolov8s, 3클래스, val 0.967 / **test 0.957**). 작성: MLOps 관점.
> 모든 지표는 **best.pt 단독 재추론으로 통일 측정**(val=자기 데이터셋, test=공통 3클래스 hold-out `data_final_backup`, conf 0.25). MLflow 캡처와 동일.

---

## 0. 한눈에 보기 — 전체 타임라인
> mAP50 = **val / test** (재추론 통일 측정). test는 공통 3클래스 hold-out 기준(round1·2는 2클래스 → 별도 축).

| 라운드 | 단계 성격 | base→data | 클래스 | train 장수 | val / test mAP50 | 이 라운드에서 "필요"를 느껴 추가한 것 |
|---|---|---|---|---|---|---|
| **1** | 2클래스 전이학습 검증 | yolov8s→data_remapped | head/helmet | 4,916 | 0.972 / 0.957 | (수동 학습) → **MLflow 실험추적 도입 계기** |
| **2** | 2클래스 파인튜닝 → **포게팅 발견** | r1→data_hat | head/helmet | 176 | **0.165 / 0.128** ⚠️ | 착시 0.99에 속을 뻔 → **측정 통일·기록(MLflow를 "사고 방지"로 도입)** |
| **3** | 3클래스 첫 시도(실패) | yolov8s→data_final | +fallen | 6,261 | **0.549 / 0.549** | 왜 떨어졌나 안 보임 → **클래스별 mAP·혼동행렬 로깅** |
| **4** | 3클래스 재학습(성공) | yolov8s→data_final | 3 | 6,261 | 0.938 / 0.863 | 재현·비교 필요 → **MLflow run 비교 정착** |
| **5** | 파인튜닝+샘플링·증강 | r4→data_sample/r5 | 3 | 8,028 | 0.969 / 0.866 | 반복 실행 번거로움 → **에폭별 지표 콜백 자동화** |
| **6** | 파인튜닝 | r5→data_sample/r6 | 3 | 8,028 | 0.967 / 0.965 | 다단계 수동연결 실수 위험 → **Airflow DAG 도입** |
| **7** | 파인튜닝(전체데이터) | r6→data_final_backup | 3 | 12,855 | 0.967 / **0.957** | 최고모델 보관·연결 필요 → **best 자동 백업(+레지스트리/Blob 과제)** |
| 8 | scratch 비교군 | yolov8s→data_final_backup | 3 | 12,855 | 0.949 / 0.894 | "전이 vs scratch" 객관 비교 → **실험 비교의 가치** |
| 9·10 | fallen 강화·동적파라미터 | r7→·r9→ | 3 | 12,855 | 0.961→0.956 / 0.939→0.935 | 자동 반복 → **DAG 동적 파라미터**, 그러나 과튜닝 → **"멈출 때"를 지표로 판단** |

> ⚠️ **round2**: hat-val(no_helmet 1클래스 45장)에선 0.99·precision 1.0로 보였으나, **2클래스 정상 평가 시 0.165**(ok를 망각). = 좁은 단일클래스 파인튜닝의 **카타스트로픽 포게팅** + 단일클래스 val 착시. 본문 [round1·2] 참고.

---

## 1. 프로젝트 배경 & 목표
- 산업현장 CCTV에서 ① **no_helmet**(안전모 미착용) ② **ok**(착용) ③ **fallen**(전도) 실시간 감지.
- 경량(YOLOv8s) + **재현 가능·자동화된 학습 파이프라인**(MLOps) 구축.

## 2. 데이터셋 & 라벨링 전략
### 2.1 클래스 정의 (왜 이렇게)
- `0 fallen` / `1 no_helmet`(맨머리·캡·모자=미착용 전부) / `2 ok`(안전모 착용).
- 안전 관점: "미착용"을 폭넓게(no_helmet) 잡는 게 핵심. fallen은 별도 안전사고.
### 2.2 라벨링·소스 (서로 다른 공개셋 → 단일 체계로 리매핑·병합)
| 소스 | 원본 라벨 | → 최종 | 라벨링/처리 |
|---|---|---|---|
| Roboflow 안전모 | head/helmet | head→no_helmet, helmet→ok | 공개 라벨 **리매핑**(`src/00_remap_labels.py`) |
| Kaggle hardhat(VOC) | head/helmet/person | head→1, helmet→2, person 제외 | **XML→YOLO 변환**(`src/05_remap_merge_all.py`) |
| Azure cap/hat | cap/hat | →no_helmet | Azure ML Studio export(COCO)→YOLO |
| **cap_labeled** | no_helmet/ok/person | →1/2, person 제외 | **직접 라벨링(Label Studio)** |
| Fallen(roboflow/laying/fall2) | Fall/laying/fall | →fallen | 공개 라벨, **pose→bbox 변환** |
- 통합: `src/05_remap_merge_all.py`(리맵+병합), `05_balance_dataset.py`(균형), `06_make_datasets.py`(샘플링), `05_verify_dataset.py`(검증).

### 2.3 ★ 라운드별 데이터 구성 (클래스별 train 이미지/객체 수)
> "객체"=박스 수, "이미지"=해당 클래스를 포함한 이미지 수.

**round1 — data_remapped (2클래스)**  train 4,916장
- no_helmet 객체 4,612 / 이미지 911 · ok 객체 13,919 / 이미지 4,503  *(ok 편중 시작)*

**round2 — data_hat (2클래스, hat 보강)**  train 176장
- no_helmet 객체 223 / 이미지 176  *(모자류 미착용 경계 보강용 소량 파인튜닝)*

**round3·4 — data_final (3클래스)**  train 6,261장
- fallen 3,450 / 3,449 · no_helmet 7,499 / 1,607 · ok 7,580 / 1,952  *(fallen 도입, 비교적 균형)*

**round5·6 — data_sample/round5·6 (3클래스, 샘플링+증강)**  train 8,028장 (둘 동일)
- fallen 3,849 / 3,848 · no_helmet 7,908 / 1,616 · ok 12,126 / 3,316

**round7 — data_final_backup (3클래스, 전체)**  train 12,855장
- fallen 3,450 / 3,449 · no_helmet 9,762 / 1,989 · ok **28,989** / 8,164  *(전체데이터, ok 68.8% 편중 — 한계로 인지)*

- (검증/테스트 분포는 위 집계 스크립트로 슬라이드에 표로 첨부 가능.)

---

## 3. MLOps 여정 — 라운드별 "필요 → 추가 → 배움" (회의록 스타일)

### [round1·2] 시작: 전이학습 검증 + **MLflow 도입** (+ 첫 함정)
- **상황**: COCO 사전학습 yolov8s로 2클래스(head/helmet) **전이학습**(r1, val 0.972/test 0.957) → hat 데이터(no_helmet 176장)로 **파인튜닝**(r2).
- **⚠️ round2의 함정 (발표 킬러 사례)**: hat-val에선 **0.99·precision 1.0**으로 "완벽"해 보였지만, **2클래스 정상 평가에선 0.165로 붕괴**(ok 클래스 ap50 0.078 = 거의 망각).
  - 원인 ①: **카타스트로픽 포게팅** — no_helmet 한 클래스만 든 좁은 데이터로 파인튜닝하니 기존 "ok" 지식을 잃음.
  - 원인 ②: **단일클래스 val 착시** — val에 no_helmet만 있으니 "전부 no_helmet"으로 찍어도 precision 1.0이 나옴.
- **느낀 필요**: "어떤 lr·epoch가 좋았는지 못 외운다 + **val 숫자만 믿으면 속는다**."
- **추가**: **MLflow Tracking**(파라미터·지표 자동 기록) + **대표성 있는 셋으로 검증**해야 한다는 자각.
- **배움**: 실험은 기억이 아니라 **기록**으로. 그리고 **좁은 데이터 파인튜닝/단일클래스 지표를 경계**하라 → **round3에서 scratch로 재시작**한 직접 계기.

### [round3] 위기: 3클래스 전환 실패 → **검증 가시성**
- **상황**: fallen 추가해 3클래스로 새로 학습 → **val 0.549 / test 0.549 급락**. 특히 **fallen은 test ap50 0.000**(전혀 못 잡음).
- **느낀 필요**: "전체 mAP만으론 무엇이 망가졌는지 모른다. 클래스별로 봐야 한다."
- **추가**: **클래스별 mAP·혼동행렬 로깅**, test셋 분리 검증.
- **배움**: 집계 지표는 거짓말한다 → **클래스 단위·홀드아웃 검증**이 필수. (round3의 정체 = fallen 0.000을 클래스별로 봐야 드러남)

### [round4] 회복: 제대로 된 baseline → **run 비교 정착**
- **상황**: 하이퍼파라미터 정비해 3클래스 재학습(lr0 0.01·warmup 3·가벼운 증강) → **val 0.938 / test 0.863**, 파인튜닝 체인의 진짜 base.
- **추가**: MLflow UI에서 **run 정렬·비교**로 "어느 설정이 최선인지" 객관 선택.
- **배움**: 좋은 baseline 하나가 이후 모든 파인튜닝의 기준이 된다.

### [round5·6] 반복: 파인튜닝 사이클 → **Airflow 자동화 도입**
- **상황**: r4→r5(샘플링+증강, val 0.969/test 0.866)→r6(val 0.967/test 0.965) **파인튜닝을 수동 반복**. 매번 base 바꿔 실행·검증·백업.
- **느낀 필요**: "단계가 늘수록 손으로 연결하다 실수(잘못된 base·데이터)·누락 발생."
- **추가**: **Airflow DAG** — `데이터확인 → 학습 → 검증(클래스별) → best 백업`을 한 트리거로, 단계 의존성·실패 재시도 자동화. + **에폭별 지표 MLflow 콜백**.
- **배움**: 반복되는 다단계 작업은 **오케스트레이션**으로. 사람은 설계, 실행은 파이프라인.

### [round7] 완성: 전체데이터 파인튜닝 → **모델 관리 필요**
- **상황**: r6→전체데이터(data_final_backup) 파인튜닝 → **val 0.967 / test 0.957** (test 클래스별 fallen 0.942/no_helmet 0.959/ok 0.970). **최종 채택**.
- **느낀 필요**: "최고 모델을 안전하게 **버전·보관·배포 연결**해야 한다(로컬만으론 유실/혼동 위험)."
- **추가(설계)**: best 자동 백업 → **다음 단계로 MLflow Model Registry + Azure Blob 업로드**(아래 6장).
- **배움**: 학습의 끝은 모델 파일이 아니라 **관리되는 모델 버전**.

### [round8] 대조군: scratch 비교 → **"왜 round7인가"의 증거**
- yolov8s에서 동일 전체데이터를 **scratch**로 학습.
- **val에선 비슷(0.949)했지만 hold-out test에서 무너짐**:
  | | val mAP50 | **test mAP50** |
  |---|---|---|
  | round7 (전이+점진 파인튜닝) | 0.967 | **0.957** |
  | round8 (scratch) | 0.949 | **0.894** |
- **결정 근거 = test 일반화**(0.957 vs 0.894, ~6%p). round7이 처음 보는 데이터에 훨씬 강함.
- **이유**: ① 전이학습이 불균형·한정 데이터에서 일반화 우위 ② 점진 파인튜닝(커리큘럼)이 안정적·소수클래스 유리 ③ scratch는 ok 68% 편중 전체데이터에 곧장 과적합.
- **교훈**: 모델은 **val이 아니라 hold-out test로** 고른다. round8은 실패가 아니라 **"전이+파인튜닝 > scratch"를 증명한 대조 실험**(MLflow 객관 비교의 가치).

### [round9·10] 과욕: 동적 파라미터 자동화 → **"멈출 때" 판단**
- **상황**: r7에서 fallen 더 끌어올리려 **DAG가 이전 라운드 클래스별 mAP를 보고 다음 라운드 증강·lr을 동적 결정**(`helmet_detection_dag.py`). 그러나 test 0.957→**0.939(r9)→0.935(r10)** **퇴보**(과튜닝).
- **배움**: 자동화가 "더 돌리기"를 쉽게 해주지만, **무작정 학습 ≠ 개선**. MLflow 지표로 **종료 기준**을 정하는 것이 MLOps의 성숙도.

---

## 4. 도구 채택의 진화 (왜 / 언제)
- **MLflow(실험관리)**: r1~ 도입. 파라미터·지표·아티팩트·**클래스별 mAP** 기록 → 19개 run 객관 비교, 최고(r7) 선택·퇴보(r9/10) 진단. 백엔드 sqlite `mlflow/mlflow.db`.
- **Airflow(오케스트레이션)**: r5~6 도입. 다단계 학습→검증→백업→**동적 파라미터** 자동화(`airflow/dags/helmet_detection_dag.py`).
- **(추가 예정) MLflow Model Registry**: r7 이후. 모델 **버전·스테이지(Staging→Production) 승격·롤백**.
- **(추가 예정) Azure Blob**: best 모델 **원격 영구 보관**(로컬+Blob 이중화).

## 5. 최종 아키텍처 = Continuous Training(CT) 파이프라인
```
[수동/사전]  데이터수집·라벨링(Label Studio)·리매핑·병합(src/00~06) · SAS발급
     │
[트리거] ──► Airflow DAG
     ├─ wait_for_gpu (게이트)
     ├─ 전이학습→파인튜닝 체인 (best.pt 자동 연결)
     ├─ 검증 (test 클래스별 mAP)
     ├─ 저장 3중: 로컬 experiments/ + 로컬백업 models/best/ + Azure Blob
     ├─ MLflow 기록(파라미터·지표·아티팩트) + Model Registry 등록/스테이지
     └─ summary
[수동/하류]  Production 승격 결정 · 추론 서비스 배포
```
- **자동화 경계**: 상류(데이터/라벨링)·하류(승격/배포)는 사람, 가운데(학습~추적~저장)는 자동 = **CT**.

## 6. MLflow Model Registry 추가 (구현 설계)
- sqlite 백엔드라 Registry 지원. 학습 후:
  ```python
  import mlflow
  from mlflow.tracking import MlflowClient
  # 1) run 아티팩트로 모델 버전 등록
  mv = mlflow.register_model(f"runs:/{run_id}/weights", "helmet-detector")
  client = MlflowClient()
  # 2) 임계값 통과 시 스테이지 승격
  if test_map50 >= 0.90:
      client.transition_model_version_stage("helmet-detector", mv.version, stage="Staging")
  # (사람 승인 후) Production 승격
  # client.transition_model_version_stage("helmet-detector", v, "Production", archive_existing_versions=True)
  ```
- **추론은 레지스트리에서 로드**: 최신 Production 버전의 `best.pt`를 받아 사용(현재 하드코딩 `MODEL_PATH` 대체).
- **이점(발표 포인트)**: 단일 진실원(SSOT)·버전·스테이지·롤백·run 계보 연결.
- DAG에 `register_model` task 추가(검증 뒤).

## 7. 구현 산출물 (승인 후)
1. `docs/mlops_journey.md` — 위 1~6장 저장(PPT 토대).
2. **round7 재현 DAG** `airflow/dags/helmet_detection_round7_repro_dag.py`:
   - round1·2(2클래스 검증) → round3(실패 재현, 선택) → **round4→5→6→7**(전이학습→파인튜닝 체인).
   - 각 단계 args.yaml 그대로(주석으로 lr/aug 명시). round4만 args 부재 → fresh 표준값 복원(명시).
   - 산출물 **전용 폴더** `experiments/round7_repro/{round1..7}/`.
   - 각 단계 best → **로컬 백업 + Azure Blob 업로드**(매 단계, 실패 시 경고만) + **MLflow Registry 등록**.
   - `schedule=None`(수동), MLOps 발표용 **한국어 주석**(단계 역할·자동화 경계 블록) 충실.
3. `.env`에 `BLOB_ACCOUNT_URL` 추가(연결 위해 필수) + Azure 업로드/Registry 코드.

## 8. 결론 & MLOps 교훈 (발표 마무리)
- 데이터·실험이 **기록**되어야 재현된다(MLflow).
- 반복 다단계는 **오케스트레이션**(Airflow)으로, 사람은 설계·판단에 집중.
- 집계 지표는 거짓말 → **클래스별·홀드아웃 검증**.
- **무작정 학습 ≠ 개선** — 지표로 멈출 때를 안다(r9/10 교훈).
- 모델의 끝은 파일이 아니라 **관리되는 버전**(Registry+원격저장).
