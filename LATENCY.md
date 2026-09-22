# 실제 카메라 지연 기록

`latency_recorder.py`는 `sender.py`의 HandTracker와 기존 JPEG/TCP 프레임 프로토콜을 재사용하는 별도 측정 프로그램입니다. 기존 sender/master 파일은 수정하지 않습니다. 측정 시 같은 카메라를 사용하는 기존 프로그램은 종료하세요.

## 실행

수신 PC 또는 마스터에서 먼저 실행합니다. receiver는 표준 라이브러리만 필요하며 여러 sender가 같은 포트에 접속할 수 있습니다.

```bash
python latency_recorder.py receiver --port 5100 --output-dir latency_logs --duration 600
```

각 카메라 Pi에서 실행합니다. 기존 카메라 프로그램과 동일한 MediaPipe/OpenCV 및 모델이 필요합니다. 아래 IP를 수신 장치 주소로 바꾸고 카메라마다 ID를 다르게 지정하세요.

```bash
python latency_recorder.py sender --host 192.168.1.100 --id 2 --fps 30 --duration 600
python latency_recorder.py sender --host 192.168.1.100 --id 3 --fps 30 --duration 600
```

CSI 카메라는 `--backend picamera2`를 추가합니다. `--camera`, `--model`, `--hands`, `--quality`도 지원합니다. `--duration`을 생략하면 Ctrl+C로 종료합니다. 재접속은 자동이며 receiver는 연결마다 새 CSV를 만듭니다. 카메라 오류는 네트워크 재접속과 별개로 해결해야 합니다. receiver는 종료 시 진행 중인 연결의 타임아웃 때문에 약 5초 더 걸릴 수 있습니다.

## 시계와 측정 범위

기본 receiver는 시계 동기화를 가정하지 않습니다. `raw_capture_to_arrival_ms`는 **수신 장치 시각 - 송신 장치 시각**이며 시계 차이가 섞여 있습니다. 이때 보정된 `capture_to_arrival_ms`와 `send_prepare_to_arrival_ms`는 빈칸입니다.

장치 시계가 필요한 정확도로 동기화되어 있음을 외부에서 확인한 경우:

```bash
python latency_recorder.py receiver --clocks-synchronized --duration 600
```

이 옵션은 시계를 동기화하거나 정확도를 검증하지 않습니다. 시간 동기화의 잔여 오차는 측정 오차로 남습니다. `clock_assumption=user_supplied_offset`, offset 0으로 기록됩니다.

알고 있는 일정한 오프셋을 직접 지정할 수도 있습니다. **offset = sender 시계 - receiver 시계**입니다. sender가 100ms 앞서 있으면 `--sender-clock-offset-ms 100`으로 지정합니다. 보정 지연은 raw 지연 + offset입니다. 한 receiver의 offset은 모든 접속에 적용되므로 카메라마다 다르면 별도 포트의 receiver를 각각 실행하세요. 시계 드리프트나 시계 점프는 이 상수로 해결되지 않습니다. 음수 지연을 0으로 숨기지 않으므로 음수 값은 시계·타임스탬프 상태를 점검하는 신호입니다.

촬영 시각은 현재 HandTracker의 **호스트 카메라 읽기 완료 직후 시각**입니다. 센서 노출 시각이 아니므로 센서/드라이버 버퍼 지연은 포함되지 않을 수 있습니다. `tracker_done_unix_ns`는 HandTracker.read 반환 직후로, 순수 추론 종료보다 좌표 메타데이터 구성 시간만큼 늦습니다. 순수 추론 구간은 기존 `inference_ms`를 사용합니다.

receiver의 도착 시각은 전체 JPEG/메타데이터 패킷을 수신하고 메타데이터 JSON을 읽은 직후입니다. 화면 디코딩/표시는 하지 않습니다. 기존 master는 디코딩·슬롯 갱신 후 ACK하지만 이 측정 receiver는 검증 후 바로 ACK하므로 기존 GUI 경로와 처리율이 다를 수 있습니다. 기존과 동일하게 한 프레임을 보내고 ACK를 기다립니다. CSV 기록 오버헤드는 후속 처리율에 영향을 줄 수 있습니다. 신뢰하는 사설 LAN에서 사용하세요.

## CSV

- receiver: `receiver_camera<ID>_<연결 UUID>.csv`
- sender: `sender_camera<ID>_<실행 UUID>.csv`
- 모든 시각 필드는 정수 ns, 간격·지연은 ms입니다. 기존 파일을 덮어쓰지 않습니다.
- 두 파일은 `camera_id`, `seq`, `capture_unix_ns`로 대응할 수 있습니다. sender와 receiver의 session ID는 서로 다른 로컬 식별자입니다.

| 열 | 의미 |
|---|---|
| capture_interval_ms | **전달된** 연속 관측의 촬영 시각 차이. 모든 센서 프레임 간격이 아님 |
| arrival_interval_ms | receiver monotonic 기준 도착 간격 |
| skipped_capture_frames | 직전 수신 이후 건너뛴 호스트 캡처 시퀀스 수. 네트워크 손실률이나 센서 누락률이 아님 |
| inference_ms | MediaPipe 추론 시간 |
| jpeg_encode_ms | JPEG 인코딩 및 바이트 변환 시간 |
| raw_capture_to_arrival_ms | 시계 차이가 포함될 수 있는 원시 차이 |
| capture_to_arrival_ms | 사용자가 지정한 시계 가정에 따른 촬영→수신 지연 |
| send_prepare_to_arrival_ms | JSON/패킷 준비 시작→수신 지연. 순수 네트워크 지연이 아님 |
| pack_ms | sender 패킷 구성 시간 |
| send_to_ack_ms | sender monotonic 기준 sendall 시작→ACK 수신. 전송·서버 처리·ACK 포함, 편도 지연이 아님 |
| cycle_ms | tracker 읽기 시작→ACK 수신. FPS 제한 대기 및 CSV 쓰기 제외 |

연결의 첫 행은 간격과 skipped 열이 빈칸입니다. ACK를 받지 못한 마지막 전송은 sender CSV에 없을 수 있고, receiver에는 이미 기록되었을 수 있습니다. FPS 옵션은 처리 상한이며 실제 30FPS를 보장하지 않습니다.

## 통계와 학습 데이터

```bash
python latency_recorder.py summarize latency_logs > latency_summary.json
```

카메라별 각 ms 지표의 개수, 평균, 모집단 표준편차, 최소, 중앙값, 95/99백분위, 최대를 JSON으로 출력합니다. 빈 지표는 집계하지 않습니다. sender와 receiver 로그를 합치면 공통 지표(추론 시간 등)가 중복 집계되므로 **한 역할의 로그만 넣은 디렉터리**를 요약하세요. 서로 다른 촬영 조건도 구분해서 보관하세요. 워밍업 구간은 원본에 보존되므로 필요 시 별도로 제외합니다.

합성 데이터에는 receiver의 실제 연속 행에서 `capture_interval_ms`와 검증된 `capture_to_arrival_ms`를 함께 샘플링하세요. 도착 지연으로 촬영 시각을 바꾸지 마세요. 전달된 간격에는 이미 건너뛴 캡처 영향이 반영되므로 같은 `skipped_capture_frames`만큼 추가 삭제하면 누락을 이중 반영합니다. 원본 시각과 연결 구간을 보존하여 혼잡 구간의 연속성을 재현할 수 있습니다. 시계가 확인되지 않았다면 raw 지연을 실제 편도 지연으로 학습에 사용하지 마세요.
