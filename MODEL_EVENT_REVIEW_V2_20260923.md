# 이벤트 모델 재평가 — 2026-09-23

평가 대상은 현재 `hand_transformer.py`, `hand_dataset.py`, 학습 코드와 `HAND_TRANSFORMER.md`다. 모델 소스나 가중치는 수정하지 않았다. 정확한 소스 SHA256과 재현 결과는 `.tools/event_model_review_v2.json`, 재현 코드는 `.tools/review_event_model_v2.py`에 있다.

## 판단

이전 평가에서 발견한 빈 이벤트 누락, 같은 도착 시각의 인과성 불일치, 과거 촬영 프레임의 최신 ray 덮어쓰기는 수정됐다. 현재 데이터셋 경로로 학습 실험을 진행할 수 있다. 다만 원시 이벤트를 직접 받는 배치 API와 스트림의 stale 이벤트 정책은 아직 다르다. 학습 후 정확도와 실제 Raspberry Pi 성능은 이번 검증 범위 밖이다.

## 확인 결과

- unittest discover: 42개 통과.
- 독립 빈 이벤트 probe: 정상 프레임과 모든 관절 미검출 프레임 모두 복원됨(2개).
- 같은 도착 시각: 이벤트를 모아 처리하거나 매번 flush해도 배치와 스트림 차이는 각각 약 0.000018 / 0.000026 mm로 수치 오차 수준. 비영 출력·보정 head를 사용해 영 초기화로 문제가 가려지지 않게 했다.
- 촬영 시각 0.10초 프레임 뒤에 0.05초 프레임이 도착해도 최신 ray는 0.10초를 유지.
- Colab notebook 내 5개 Python 소스가 현재 로컬 파일과 일치.
- 현재 기본값: dim=96, heads=4, decoder blocks=2, event encoder=1, 파라미터 763,693개. 종전 dim=128/blocks=3의 1,746,765개 대비 약 56.3% 감소. 정확도 유지 여부는 별도 학습 비교가 필요하다.

## 남은 발견: P2 — 직접 배치 입력에서 stale 이벤트 제거 정책 불일치

위치: `hand_transformer.py:252`, `hand_transformer.py:612`, `hand_transformer.py:668`.

`EventStream.push()`는 같은 카메라의 직전 수락 촬영 시각 이하인 이벤트를 버린다. 반면 `forward()`와 `make_events()`는 해당 이벤트를 그대로 유지한다. `latest_rays()`가 최신 촬영 ray를 고르더라도 stale 이벤트 자체는 calibration evidence와 decoder attention에 참여하므로 두 경로의 예측이 달라진다.

독립 합성 재현에서 마지막 stale 이벤트를 포함한 배치와 스트림의 관절 평균 차이는 **1.618 mm**였다. 배치에서도 스트림이 수락한 이벤트만 남기면 **0.000017 mm**로 줄었다. 이 값은 무작위 비영 head와 world unit 30 cm를 사용한 일관성 검사이며, 학습된 모델의 정확도 수치가 아니다.

현재 `clip_events()`는 시뮬레이터의 selected_frame으로 복원하므로 stale 이벤트가 선택되지 않아 표준 학습 경로의 즉각적 장애는 아니다. 원시 패킷을 직접 배치 평가하거나 다른 로더를 붙일 때 재현된다.

권장 수정: arrival 순서에서 카메라별 이전 최대 capture를 초과한 이벤트만 수락하는 공통 정책을 둔다. 배치에서는 수락 여부를 event_present에 반영해 padding 형태와 인덱스를 보존하고, 이를 스트림 정책과 일치시킨다. 또는 forward가 이미 필터된 이벤트만 받는다는 전제조건을 명시하고 위반을 검출한다. 늦은 프레임·중복 capture를 포함한 배치/스트림 동등성 회귀검사가 필요하다.

## 실행 성능과 검증 한계

현재 PC CPU, PyTorch 2 threads, 기본 모델 eval/inference mode, 합성 3-camera 17.1Hz 입력에서 매 query마다 새 이벤트 3개 push와 query를 함께 측정했다. 첫 10 query를 제외한 42회 결과는 평균 **9.10 ms**, 중앙값 **9.01 ms**, p95 **11.30 ms**다. 이전 실행 평균은 9.55 ms였다. 카메라 획득, 2D detector, 전송, 화면 표시를 제외한 3D 모델 처리 시간이며, 전체 시스템 FPS로 환산하면 안 된다.

이벤트별 decoder key/value 캐시와 pending 이벤트 일괄 처리는 타당한 최적화이며 정상 입력의 결과 일관성도 확인됐다. 문서의 Pi 5 30–55 ms는 추정이며 실제 장치에서 검증하지 않았다. 15 FPS라면 전체 프레임 간격은 약 66.7 ms지만, 2D detector의 60 ms와 이번 모델 시간을 단순 합산할 수 있는지는 실행 장치와 병렬 파이프라인 구성에 달려 있다.

학습된 새 소형 모델의 held-out MPJPE, 빠른 움직임·가림·오검출별 성능, 기존 크기 대비 성능은 이번에 측정하지 않았다. 이전 anchor-only 수치나 smoke 체크포인트를 본학습 정확도로 해석하면 안 된다. 다음 판단 근거는 같은 데이터·학습 조건에서 96/2와 128/3의 validation 결과, 그리고 목표 장치에서 전체 파이프라인 latency 측정이다.
