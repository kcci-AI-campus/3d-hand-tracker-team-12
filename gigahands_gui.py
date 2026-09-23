"""Offline Tk GUI with a rotatable 3D skeleton and three image-plane previews."""
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from gigahands_sim import Config, demo_motion, load_motion, simulate, save_dataset, default_error_ranges
from motion_archive import is_motion_archive, list_motion_members


def camera_frame(origin, rotation, intrinsics, width, height, depth=.45):
    """Image-plane corners in world space; rotation maps world to camera."""
    fx, fy, cx, cy = intrinsics
    uv = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    local = np.column_stack(((uv[:, 0]-cx)/fx, (uv[:, 1]-cy)/fy, np.ones(4))) * depth
    return np.asarray(origin) + local @ rotation


PARAMETER_HELP = {
    'timing_profile': '기본 profiles/pi_c270_timing.json. 실측 촬영 간격·처리 대기·전송 지연을 같은 행으로 재생합니다. 카메라마다 시작 행을 랜덤 선택합니다. 이 모드에서는 camera_fps, capture_jitter_ms, processing_ms, latency_ms, latency_jitter_ms, rolling_shutter_ms 수동값을 무시합니다. 빈칸이면 수식 모드입니다.',
    'world_unit_cm': '1 월드 단위의 실제 길이(cm)입니다. 기본 30이면 (-1,1) 공간은 축마다 -30~30cm, 전체 60cm입니다. 좌표 숫자는 유지하며 물리 단위를 정의합니다. 원본 단위 변환은 scale로 별도 지정하세요.',
    'position_limit_cm': '카메라 위치 오차 벡터의 전체 길이 상한(cm)입니다. 기본 3cm = 0.1 월드 단위. 축별 상한이 아닙니다. 초과한 Gaussian 표본은 다시 뽑습니다. 0이면 위치 오차가 없습니다.',
    'angle_limit_deg': '카메라의 실제 상대 회전각 상한(도)입니다. 기본 10°. 회전벡터 전체 길이를 제한하며 초과 표본은 다시 뽑습니다. 0이면 방향 오차가 없습니다.',
    'randomize_errors': '기본 true. 오차 강도·확률을 error_ranges에서 시퀀스마다 뽑습니다. 동일 seed는 동일 결과입니다. 화각·해상도·명목 내부 파라미터는 고정됩니다. false면 수동 오차값을 사용하지만 timing_profile의 시간 재생은 유지됩니다.',
    'error_ranges': '랜덤화할 변수별 [최솟값,최댓값] JSON입니다. 범위 편집 버튼으로 수정하세요. 위치·각도 범위는 오차 자체의 상한이 아니라 Gaussian 표준편차를 뽑는 범위입니다. 포함된 변수의 수동 값은 랜덤 모드에서 사용하지 않습니다.',
    'source_fps': '원본 애니메이션의 초당 프레임 수입니다. 파일에 timestamps가 없을 때만 사용합니다. 실제 원본 FPS에 맞추세요. 데모는 30 FPS입니다.',
    'camera_fps': '수식 모드에서 사용할 관측 FPS입니다. 기본 17.1은 실측 송신 관측률에 가깝게 설정한 값입니다. timing_profile을 사용하면 이 값 대신 실측 촬영 간격을 재생합니다. 원본 애니메이션 FPS는 source_fps로 따로 지정합니다.',
    'output_fps': '학습 정답을 생성하는 초당 예측 시점 수입니다. 각 시점까지 도착한 관측으로 그 시점의 3D 자세를 예측합니다.',
    'scale': '원본 좌표를 월드 단위로 바꾸는 배율입니다. fit_extent=0일 때 적용됩니다. 1월드=30cm 기준 원본 mm는 1/300(약 0.00333333), cm는 1/30, m는 1/0.3을 사용합니다. 카메라 위치에는 적용하지 않습니다.',
    'center': 'true이면 시퀀스 전체의 좌표 최솟값·최댓값으로 구한 중심을 원점으로 옮깁니다. 모든 프레임에 같은 이동을 적용하므로 손의 이동은 유지됩니다.',
    'fit_extent': '0보다 크면 가장 먼 좌표의 절댓값이 이 값이 되도록 시퀀스 전체를 균일 확대·축소합니다. 범위: 0 이상 1 미만. 0이면 scale 사용. 자동 맞춤은 실제 손 크기를 바꿉니다.',
    'axis_order': '원본 손 좌표의 축을 읽는 순서입니다. xyz는 그대로, xzy는 원본 Y/Z 교환입니다. 카메라 위치나 월드 좌표계 정의에는 적용되지 않습니다.',
    'axis_sign': '축 재배열 후 손 좌표의 각 축에 곱할 부호입니다. [1,1,1]은 그대로, [-1,1,1]은 X 반전입니다. 각 값은 1 또는 -1입니다.',
    'joint_order': '출력 관절 0~20에 대응하는 원본 관절 인덱스입니다. 0~20을 한 번씩 포함하는 JSON 배열. 뷰어 순서는 손목, 엄지, 검지, 중지, 약지, 소지입니다.',
    'swap_hands': 'true이면 원본의 두 손 순서를 서로 바꿉니다. 좌우 손을 자동 판별하지 않으므로 원본 hand 0/1의 의미를 먼저 확인하세요.',
    'seed': '0 이상의 정수 난수 시드입니다. 같은 입력·설정·시드를 사용하면 같은 오차와 지연을 재현합니다. 다른 시드로 증강 변형을 만듭니다.',
    'window': 'Transformer 입력으로 묶는 연속 예측 프레임 수입니다. 양의 정수. 16이면 최대 16×3×2×21=2016개 토큰이며 정답은 마지막 시점의 두 손 자세입니다.',
    'stride': '연속 학습 윈도의 시작점을 몇 출력 프레임씩 이동할지 지정합니다. 양의 정수. 1이면 한 프레임씩 이동하여 윈도가 많이 겹칩니다.',
    'cameras': '세 카메라의 명목 월드 위치 [x,y,z]입니다. 기본값은 [[-1,-1,0.5],[1,-1,0.5],[0,1,1]]. 1월드=30cm에서 실제 위치는 (-30,-30,15), (30,-30,15), (0,30,30)cm입니다.',
    'targets': '각 카메라가 바라보는 월드 좌표 [x,y,z] 3개입니다. [[0,0,0],[0,0,0],[0,0,0]]은 모두 원점을 바라봅니다. 카메라 위치와 같은 점은 지정할 수 없습니다.',
    'width': '투영 영상의 가로 해상도입니다. 단위: 픽셀, 양의 정수. 기본 320. 실측 intrinsics를 쓰면 이 해상도에 맞춘 보정값을 함께 입력하세요.',
    'height': '투영 영상의 세로 해상도입니다. 단위: 픽셀, 양의 정수. 기본 240. 영상 v 좌표는 아래쪽으로 증가합니다.',
    'diagonal_fov': '대각 화각입니다. 단위: 도, 1 초과 170 미만. intrinsics가 비어 있을 때 초점 거리를 추정합니다. 기본 55°는 C270 초기 추정값이며 320×240 실측 보정값은 아닙니다.',
    'intrinsics': '카메라별 [fx,fy,cx,cy] 배열 3개입니다. 모두 픽셀 단위. fx/fy는 초점 거리, cx/cy는 주점입니다. []이면 화각으로 추정합니다. 값을 입력하면 diagonal_fov보다 우선합니다.',
    'position_std': '카메라별 위치 오차를 뽑는 축별 Gaussian 표준편차(월드 단위)입니다. 1월드=30cm에서 0.033333은 1cm입니다. 벡터 전체 오차가 position_limit_cm를 넘으면 다시 뽑습니다. 0이면 위치 오차가 없습니다.',
    'angle_std_deg': '카메라별 yaw·pitch 오차의 Gaussian 표준편차(도)입니다. Roll은 항상 0이며 카메라 화면의 수평을 유지합니다. 설정 파일에 각도 상한이 있으면 합성 회전각을 제한합니다. 시퀀스 동안 고정됩니다.',
    'focal_std_pct': '카메라별 fx/fy 오차 크기입니다. 입력값/100을 로그 배율의 표준편차로 사용합니다. 작은 값에서는 대략 퍼센트 오차이며 1은 약 1%입니다. 시퀀스 동안 고정됩니다.',
    'principal_std_px': '카메라별 주점 cx/cy 오차의 Gaussian 표준편차입니다. 단위: 픽셀. 시퀀스 동안 고정되며 ray 계산에는 명목 주점을 사용합니다.',
    'k1': '실제 투영의 1차 방사 왜곡 계수입니다. 무차원. 배율은 1+k1·r²+k2·r⁴입니다. ray는 왜곡 없는 명목 모델을 사용하므로 잔여 보정 오차를 만듭니다. 0은 이 항을 끕니다.',
    'k2': '실제 투영의 2차 방사 왜곡 계수입니다. 무차원. k2·r⁴ 항으로 영상 중심에서 먼 점에 더 큰 영향을 줍니다. k1=k2=0이면 렌즈 왜곡이 없습니다.',
    'pixel_std': '검출된 각 관절의 u/v 좌표에 추가하는 독립 Gaussian 노이즈의 표준편차입니다. 단위: 픽셀. 1이면 각 축에 표준편차 1px의 흔들림을 적용합니다.',
    'outlier_prob': '관절마다 큰 검출 오차가 추가될 확률입니다. 범위: 0~1. 0.01은 1%이며 크기는 outlier_std_px로 설정합니다.',
    'outlier_std_px': '이상치로 선택된 관절에 추가하는 Gaussian 오차의 표준편차입니다. 단위: 픽셀. 일반 pixel_std 노이즈에 더해집니다.',
    'missing_prob': '각 관절 관측을 무작위로 누락할 확률입니다. 범위: 0~1. 0.02는 2%. 누락된 입력은 0으로 채우고 마스크를 false로 저장합니다. 실제 표면 가림을 계산하는 것은 아닙니다.',
    'packet_loss': '카메라 프레임 전체가 통신 중 손실될 확률입니다. 범위: 0~1. 양손 관측이 함께 손실되며 기존에 도착한 프레임은 max_age_ms까지 사용할 수 있습니다.',
    'latency_ms': '통신 지연을 샘플링하는 Gaussian 분포의 평균 파라미터입니다. 단위: ms. 음수 샘플은 0으로 제한하므로 실제 평균은 달라질 수 있습니다. 처리·롤링셔터 시간은 별도로 더합니다.',
    'latency_jitter_ms': '프레임마다 달라지는 통신 지연의 표준편차입니다. 단위: ms. 클수록 도착 간격이 불규칙해지고 프레임 도착 순서가 뒤바뀔 수 있습니다.',
    'processing_ms': '각 프레임의 고정 추론·처리 시간입니다. 단위: ms. 통신 지연과 별개로 도착 시각에 더해집니다.',
    'capture_jitter_ms': '예정된 촬영 시각에 추가하는 Gaussian 시간 오차의 표준편차입니다. 단위: ms. 촬영 간격의 흔들림이며 통신 지연과는 다릅니다.',
    'clock_offset_std_ms': '카메라 시계의 고정 시간 오프셋을 뽑는 표준편차입니다. 단위: ms. 보고 촬영 타임스탬프에만 영향을 주며 실제 촬영·도착 시각은 바꾸지 않습니다.',
    'clock_drift_std_ppm': '카메라 시계 속도 오차의 표준편차입니다. 단위: ppm. 100ppm은 1초당 약 0.1ms 수준의 시간 차이에 해당합니다. 보고 촬영 시각에 누적됩니다.',
    'rolling_shutter_ms': '영상 첫 행부터 마지막 행까지 읽는 시간입니다. 단위: ms. 행 위치에 따라 다른 시점의 손 자세를 보간하는 근사이며 도착 시각에도 읽기 시간을 더합니다. 0이면 비활성화됩니다.',
    'max_age_ms': '예측 시점에서 관측의 실제 촬영 시점까지 허용하는 최대 나이입니다. 단위: ms, 0보다 커야 합니다. 초과한 관측은 마스크 false로 처리합니다. 시계 오차 없는 실제 시각 기준입니다.',
}


class ParameterTooltip:
    def __init__(self, widget, text, on_select):
        self.widget, self.text, self.on_select = widget, text, on_select
        self.popup = None
        self.pending = None
        widget.bind('<Enter>', self.enter, add='+')
        widget.bind('<Leave>', self.hide, add='+')
        widget.bind('<FocusIn>', lambda event: on_select(), add='+')
        widget.bind('<ButtonPress>', self.hide, add='+')
        widget.bind('<Destroy>', self.hide, add='+')

    def enter(self, event=None):
        self.hide()
        self.on_select()
        self.pending = self.widget.after(450, self.show)

    def show(self):
        self.pending = None
        self.popup = tk.Toplevel(self.widget)
        self.popup.withdraw()
        self.popup.wm_overrideredirect(True)
        ttk.Label(self.popup, text=self.text, wraplength=390, padding=12,
                  relief='solid', borderwidth=1).pack()
        self.popup.update_idletasks()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth()-self.popup.winfo_reqwidth()-10)
        y = self.widget.winfo_rooty()+self.widget.winfo_height()+5
        if y+self.popup.winfo_reqheight() > self.widget.winfo_screenheight():
            y = self.widget.winfo_rooty()-self.popup.winfo_reqheight()-5
        self.popup.geometry(f'+{max(0,x)}+{max(0,y)}')
        self.popup.deiconify()

    def hide(self, event=None):
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.popup is not None:
            popup, self.popup = self.popup, None
            popup.destroy()


class App:
    def __init__(self, root, cfg, path=None, member=None):
        self.root = root
        root.title('손 동작 데이터 생성기')
        root.geometry('1440x900'); root.minsize(1050, 700)
        self.cfg = cfg; self.path = path; self.member = member; self.result = None; self.busy = False
        self.motion_cache = None
        self.motion_caches = OrderedDict()
        self.archive_cache = {}
        self.archive_path = None
        self.resume_frame = 0
        self.events = queue.Queue(); self.playing = False; self.frame = 0
        self.yaw = .65; self.pitch = .3; self.zoom = 1.; self.drag = None
        self.vars = {name: tk.StringVar(value=self.format(value)) for name, value in asdict(cfg).items()}
        toolbar = ttk.Frame(root, padding=8); toolbar.pack(fill='x')
        for title, action in [('Load Motion', self.load), ('Demo', self.demo),
                              ('Load Config', self.load_config), ('Save Config', self.save_config),
                              ('Apply', self.generate), ('Export NPZ', self.export),
                              ('NPZ Reader', self.open_reader)]:
            ttk.Button(toolbar, text=title, command=action).pack(side='left', padx=3)
        browser_bar = ttk.Frame(root, padding=(8, 0, 8, 6)); browser_bar.pack(fill='x')
        self.archive_button = ttk.Button(browser_bar, text='Browse Clips', command=self.reopen_archive, state='disabled')
        self.archive_button.pack(side='left', padx=3)
        ttk.Button(browser_bar, text='Randomize Pose', command=self.randomize_pose).pack(side='left', padx=3)
        self.archive_label = tk.StringVar(value='압축 파일을 불러오면 동작 목록을 다시 열 수 있습니다.')
        ttk.Label(browser_bar, textvariable=self.archive_label).pack(side='left', padx=10)
        self.status = tk.StringVar(value='준비')
        ttk.Label(root, textvariable=self.status, padding=6).pack(fill='x', side='bottom')
        body = ttk.Panedwindow(root, orient='horizontal'); body.pack(fill='both', expand=True)
        settings = ttk.Frame(body, width=340); body.add(settings, weight=0)
        self.parameter_help = tk.StringVar(value='입력칸을 선택하면 설명이 표시됩니다.')
        help_panel = ttk.LabelFrame(settings, text='도움말', padding=10)
        help_panel.pack(side='bottom', fill='x', padx=8, pady=8)
        ttk.Label(help_panel, textvariable=self.parameter_help, wraplength=300,
                  justify='left').pack(fill='x')
        self.tooltips = []
        tabs = ttk.Notebook(settings); tabs.pack(fill='both', expand=True, padx=8, pady=8)
        def field(parent, row, label, var, help_text):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=8)
            entry = ttk.Entry(parent, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky='e', padx=8)
            self.tooltips.append(ParameterTooltip(entry, help_text,
                lambda text=help_text: self.parameter_help.set(text)))
            return entry
        setup = ttk.Frame(tabs, padding=12); tabs.add(setup, text='카메라')
        ttk.Label(setup, text='카메라 위치 (cm)').grid(row=0, column=0, columnspan=4, sticky='w', pady=8)
        for axis, label in enumerate(['X', 'Y', 'Z']):
            ttk.Label(setup, text=label).grid(row=1, column=axis+1)
        self.camera_vars = [[tk.StringVar() for _ in range(3)] for _ in range(3)]
        for c in range(3):
            ttk.Label(setup, text=f'카메라 {c+1}').grid(row=c+2, column=0, padx=4, pady=8)
            for axis in range(3):
                ttk.Entry(setup, textvariable=self.camera_vars[c][axis], width=7).grid(row=c+2, column=axis+1, padx=3)
        self.camera_summary = tk.StringVar()
        ttk.Label(setup, textvariable=self.camera_summary, wraplength=290, justify='left').grid(row=5, column=0, columnspan=4, sticky='w', pady=16)
        noise = ttk.Frame(tabs, padding=12); tabs.add(noise, text='설치 오차')
        ttk.Label(noise, text='설치 오차의 표준편차 범위', font=('', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=8)
        self.pose_vars = {name: [tk.StringVar(), tk.StringVar()] for name in ('position_std', 'angle_std_deg')}
        for row, (name, label, unit) in enumerate([('position_std', '위치', 'cm'), ('angle_std_deg', 'Yaw·Pitch', '°')]):
            for bound, text in enumerate(['최소', '최대']):
                field(noise, 1+row*2+bound, f'{label} {text} ({unit})', self.pose_vars[name][bound],
                      '클립마다 이 범위에서 축별 표준편차를 선택합니다. 상한 없이 정규분포로 뽑습니다. 최소와 최대가 같으면 표준편차가 고정됩니다. 클립 안에서는 위치·각도 오차가 유지됩니다.')
        ttk.Label(noise, text='Roll 0° 고정 · Yaw/Pitch만 무작위 · 상한 없음\n최소=최대이면 표준편차 고정, 실제 오차는 매번 추출합니다.', wraplength=290).grid(row=7, column=0, columnspan=2, sticky='w', pady=16)
        data = ttk.Frame(tabs, padding=12); tabs.add(data, text='학습 데이터')
        for row, (name, label) in enumerate([('source_fps', '원본 FPS'), ('output_fps', '출력 FPS'),
                                            ('window', '입력 프레임 수'), ('stride', '샘플 간격 (프레임)'), ('seed', '난수 번호')]):
            field(data, row, label, self.vars[name], PARAMETER_HELP[name])
        self.processing_summary = tk.StringVar()
        ttk.Label(data, textvariable=self.processing_summary, wraplength=290, justify='left').grid(row=5, column=0, columnspan=2, sticky='w', pady=16)
        timing = ttk.Frame(tabs, padding=12); tabs.add(timing, text='프레임 지연')
        self.timing_mode = tk.StringVar()
        ttk.Label(timing, text='지연 생성 방식').grid(row=0, column=0, sticky='w', pady=8)
        mode_box = ttk.Combobox(timing, textvariable=self.timing_mode, values=['실측 재생', '수동 설정'], state='readonly', width=12)
        mode_box.grid(row=0, column=1, padx=8)
        mode_box.bind('<<ComboboxSelected>>', lambda e: self.update_timing_controls())
        self.timing_entries = []
        for row, (name, label) in enumerate([('camera_fps', '카메라 FPS'), ('processing_ms', '송신 전 처리 (ms)'),
                                           ('latency_ms', '전송 지연 평균 (ms)'), ('latency_jitter_ms', '전송 지연 표준편차 (ms)'),
                                           ('capture_jitter_ms', '촬영 간격 지터 (ms)')], 1):
            self.timing_entries.append(field(timing, row, label, self.vars[name], PARAMETER_HELP[name]))
        field(timing, 6, '관측 유효 시간 (ms)', self.vars['max_age_ms'], PARAMETER_HELP['max_age_ms'])
        self.timing_hint = tk.StringVar()
        ttk.Label(timing, textvariable=self.timing_hint, wraplength=300, justify='left').grid(row=7, column=0, columnspan=2, sticky='w', pady=10)
        self.timing_summary = tk.StringVar(value='적용 후 사용된 지연을 표시합니다.')
        ttk.Label(timing, textvariable=self.timing_summary, wraplength=300, justify='left').grid(row=8, column=0, columnspan=2, sticky='w', pady=8)
        self.sync_form(cfg)
        view = ttk.Frame(body); body.add(view, weight=1)
        ttk.Label(view, text='카메라: 회색 점선 = 명목 / 분홍 실선 = 실제 (실제 크기) • 드래그 회전 / 휠 확대\n손 = 3D 정답 • 손목 ray = 명목 보정값으로 계산한 모델 입력', padding=6).pack(fill='x')
        self.pose_info = tk.StringVar()
        ttk.Label(view, textvariable=self.pose_info, padding=6, justify='left', wraplength=650).pack(fill='x')
        self.canvas = tk.Canvas(view, bg='#111827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.draw())
        self.canvas.bind('<ButtonPress-1>', lambda e: setattr(self, 'drag', (e.x, e.y)))
        self.canvas.bind('<B1-Motion>', self.orbit)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.previews = tk.Canvas(view, height=225, bg='#182235', highlightthickness=0)
        self.previews.pack(fill='x'); self.previews.bind('<Configure>', lambda e: self.draw())
        controls = ttk.Frame(view, padding=6); controls.pack(fill='x')
        ttk.Button(controls, text='Play / Pause', command=self.toggle).pack(side='left')
        self.slider = ttk.Scale(controls, from_=0, to=1, command=self.scrub)
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        self.ray_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='Rays', variable=self.ray_var, command=self.draw).pack(side='left')
        self.info = tk.StringVar(); ttk.Label(view, textvariable=self.info, padding=6).pack(fill='x')
        self.points, self.times = demo_motion()
        if path and is_motion_archive(path) and member is None:
            self.scan_archive(path)
        else:
            self.generate()
        root.after(30, self.tick)

    @staticmethod
    def format(value):
        return value if isinstance(value, str) else json.dumps(value)

    def open_reader(self):
        from npz_reader import Reader
        Reader(tk.Toplevel(self.root))

    def sync_form(self, cfg):
        for c in range(3):
            for axis in range(3):
                self.camera_vars[c][axis].set(f'{cfg.cameras[c][axis]*cfg.world_unit_cm:g}')
        for name, variables in self.pose_vars.items():
            bounds = cfg.error_ranges.get(name, [getattr(cfg, name)]*2) if cfg.randomize_errors else [getattr(cfg, name)]*2
            factor = cfg.world_unit_cm if name == 'position_std' else 1
            for var, value in zip(variables, bounds): var.set(f'{value*factor:.10g}')
        target_text = '세 카메라는 원점을 바라봅니다.' if np.allclose(cfg.targets, 0) else '시선 방향: 불러온 설정 사용'
        self.camera_summary.set(f'{target_text}\n해상도 {cfg.width} × {cfg.height} · 대각 화각 {cfg.diagonal_fov:g}°\n1 월드 단위 = {cfg.world_unit_cm:g} cm')
        timing = '실측 촬영 간격과 지연 사용' if cfg.timing_profile else '설정 파일의 시간 모델 사용'
        self.processing_summary.set(f'{timing}\n프레임 지연 탭에서 확인·변경할 수 있습니다.')
        self.timing_mode.set('실측 재생' if cfg.timing_profile else '수동 설정')
        if not cfg.timing_profile and cfg.randomize_errors:
            for name in ('processing_ms', 'latency_ms', 'latency_jitter_ms', 'capture_jitter_ms'):
                if name in cfg.error_ranges:
                    self.vars[name].set(self.format(sum(cfg.error_ranges[name])/2))
        self.update_timing_controls()

    def update_timing_controls(self):
        measured = self.timing_mode.get() == '실측 재생'
        for entry in self.timing_entries:
            entry.configure(state='disabled' if measured else 'normal')
        self.timing_hint.set('실측 촬영 간격·처리·전송 지연을 함께 재생합니다. 위 수동값은 적용하지 않습니다.\n변경 후 Apply를 누르세요.' if measured else
                             '총 지연 = 송신 전 처리 + 매 프레임 전송 지연. 전송 지연은 정규분포로 추출하고 음수는 0으로 처리합니다.\n변경 후 Apply를 누르세요.')

    def update_timing_summary(self):
        meta = json.loads(self.result['metadata'])
        cfg = meta['config']
        replay = meta.get('timing_replay')
        if replay:
            summary = replay.get('summary') or {}
            total = summary.get('capture_to_arrival_ms', {})
            transport = summary.get('send_prepare_to_arrival_ms', {})
            lines = ['적용됨: 실측 재생', Path(replay['path']).name]
            if total:
                lines.append(f'측정 총 지연: 평균 {total["mean"]:.1f} / P95 {total["p95"]:.1f} ms')
            if transport:
                lines.append(f'측정 전송 구간: 평균 {transport["mean"]:.2f} ms')
            lines.append('총 지연에는 처리·대기 시간이 포함됩니다. 노출 시작 기준은 아닙니다.')
        else:
            lines = ['적용됨: 수동 설정', f'처리 {cfg["processing_ms"]:g} ms + 전송 평균 {cfg["latency_ms"]:g} ms',
                     f'전송 표준편차 {cfg["latency_jitter_ms"]:g} ms / {cfg["camera_fps"]:g} FPS']
        lines.append(f'관측 유효 시간: {cfg["max_age_ms"]:g} ms')
        self.timing_summary.set('\n'.join(lines))

    def read_config(self):
        defaults = asdict(Config()); values = {}
        for name, var in self.vars.items():
            try:
                values[name] = var.get() if isinstance(defaults[name], str) else json.loads(var.get())
            except ValueError as exc:
                raise ValueError(f'{name}: invalid JSON value') from exc
        cfg = Config(**values)
        cfg.position_limit_cm = None
        cfg.angle_limit_deg = None
        cfg.cameras = [[float(var.get())/cfg.world_unit_cm for var in row] for row in self.camera_vars]
        cfg.error_ranges = dict(cfg.error_ranges)
        # Preserve manual values for other errors when loading an older manual config.
        if not cfg.randomize_errors:
            cfg.error_ranges = {name: [getattr(cfg, name)]*2 for name in default_error_ranges()}
        cfg.randomize_errors = True
        for name, variables in self.pose_vars.items():
            factor = cfg.world_unit_cm if name == 'position_std' else 1
            cfg.error_ranges[name] = [float(var.get())/factor for var in variables]
        if self.timing_mode.get() == '실측 재생':
            cfg.timing_profile = cfg.timing_profile or Config().timing_profile
        else:
            cfg.timing_profile = ''
            for name in ('processing_ms', 'latency_ms', 'latency_jitter_ms', 'capture_jitter_ms'):
                cfg.error_ranges.pop(name, None)
            cfg.rolling_shutter_ms = 0
            cfg.error_ranges.pop('rolling_shutter_ms', None)
        cfg.validate(); return cfg

    def load(self):
        if self.busy: return
        path = filedialog.askopenfilename(filetypes=[('Motion / Archive', '*.json *.npy *.npz *.tar.gz *.tgz *.tar'), ('All files', '*.*')])
        if path:
            if is_motion_archive(path): self.scan_archive(path)
            else:
                self.path = path; self.member = None
                self.generate()

    def reopen_archive(self):
        if not self.busy and self.archive_path:
            self.scan_archive(self.archive_path)

    def randomize_pose(self):
        if self.busy or self.result is None: return
        try:
            cfg = self.read_config()
        except Exception as exc:
            messagebox.showerror('설정 오류', str(exc)); return
        self.vars['pose_seed'].set(str((cfg.pose_seed if cfg.pose_seed is not None else cfg.seed) + 1))
        self.generate(keep_frame=True)

    def scan_archive(self, path):
        if self.busy: return
        try:
            stat = Path(path).stat()
            key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
        except OSError as exc:
            messagebox.showerror('압축 파일 오류', str(exc)); return
        if key in self.archive_cache:
            self.select_archive_member(path, self.archive_cache[key])
            return
        self.busy = True; self.playing = False
        self.status.set('압축 내부 목록을 읽는 중… 대용량 tar.gz는 전체 압축 스트림 검사에 시간이 걸립니다.')
        def work():
            try: self.events.put(('archive', (path, key, list_motion_members(path))))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def select_archive_member(self, path, members):
        self.archive_path = path
        self.archive_button.configure(state='normal')
        self.archive_label.set(f'{Path(path).name} · 동작 {len(members):,}개')
        dialog = tk.Toplevel(self.root); dialog.title('압축 안의 3D 시퀀스 선택'); dialog.geometry('900x550')
        ttk.Label(dialog, text=f'{len(members):,}개 후보 · keypoints_3d_mano 우선 · 이름으로 검색', padding=8).pack(anchor='w')
        search = tk.StringVar(); entry = ttk.Entry(dialog, textvariable=search); entry.pack(fill='x', padx=8)
        panel = ttk.Frame(dialog); panel.pack(fill='both', expand=True, padx=8, pady=8)
        listing = tk.Listbox(panel, exportselection=False)
        scroll = ttk.Scrollbar(panel, command=listing.yview); listing.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); listing.pack(fill='both', expand=True)
        visible = []
        def refresh(*args):
            visible[:] = [name for name in members if search.get().casefold() in name.casefold()]
            listing.delete(0, 'end')
            for name in visible: listing.insert('end', name)
            if visible:
                index = visible.index(self.member) if path == self.path and self.member in visible else 0
                listing.selection_set(index); listing.see(index)
        def choose(event=None):
            selection = listing.curselection()
            if not selection: return
            self.path = path; self.member = visible[selection[0]]
            dialog.destroy(); self.generate()
        search.trace_add('write', refresh); refresh()
        listing.bind('<Double-Button-1>', choose)
        ttk.Button(dialog, text='Load Clip', command=choose).pack(pady=8)
        dialog.transient(self.root); dialog.grab_set(); entry.focus_set()
        self.status.set('압축 안의 시퀀스를 선택하세요.')

    def demo(self):
        if self.busy: return
        self.points, self.times = demo_motion(); self.path = None; self.member = None; self.generate()

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[('Config', '*.json')])
        if path:
            try:
                with open(path, encoding='utf-8') as f: cfg = Config(**json.load(f))
                cfg.validate()
                for name, value in asdict(cfg).items(): self.vars[name].set(self.format(value))
                self.sync_form(cfg)
                self.status.set('설정을 불러왔습니다. 적용을 누르면 미리보기가 갱신됩니다.')
            except Exception as exc: messagebox.showerror('Config error', str(exc))

    def save_config(self):
        try:
            cfg = self.read_config()
            path = filedialog.asksaveasfilename(defaultextension='.json')
            if path:
                with open(path, 'w', encoding='utf-8') as f: json.dump(asdict(cfg), f, indent=2)
        except Exception as exc: messagebox.showerror('Config error', str(exc))

    def generate(self, keep_frame=False):
        if self.busy: return
        try:
            cfg = self.read_config()
        except Exception as exc:
            messagebox.showerror('Settings error', str(exc)); return
        self.resume_frame = self.frame if keep_frame else 0
        self.busy = True; self.playing = False; self.result = None
        self.status.set('시퀀스 로드 및 시뮬레이션 중…'); self.draw()
        points, times = self.points.copy(), self.times.copy()
        path, member = self.path, self.member
        cached_motions = dict(self.motion_caches)
        source = f'{path}::{member}' if member else path or 'synthetic demo'
        def work():
            try:
                from pathlib import Path
                stat = Path(path).stat() if path else None
                cache_key = (str(path), member, cfg.source_fps, stat.st_size, stat.st_mtime_ns) if stat else None
                if cache_key is not None and cache_key in cached_motions:
                    p, t = cached_motions[cache_key]
                else:
                    p, t = load_motion(path, cfg.source_fps, member) if path else (points, times)
                cache = (cache_key, p, t) if cache_key is not None else None
                self.events.put(('result', (simulate(p, t, cfg), cfg, source, cache)))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def export(self):
        if self.busy or self.result is None: return
        path = filedialog.asksaveasfilename(defaultextension='.npz', filetypes=[('Dataset', '*.npz')])
        if path:
            self.busy = True; self.status.set('데이터 저장 중…')
            result, source = self.result, self.source
            def work():
                try:
                    save_dataset(path, result, source); self.events.put(('saved', path))
                except Exception as exc: self.events.put(('error', str(exc)))
            threading.Thread(target=work, daemon=True).start()

    def toggle(self):
        self.playing = not self.playing; self.last_tick = time.monotonic()

    def scrub(self, value):
        self.frame = int(float(value)); self.draw()

    def orbit(self, event):
        if self.drag:
            self.yaw += (event.x-self.drag[0])*.008
            self.pitch = float(np.clip(self.pitch+(event.y-self.drag[1])*.008, -1.5, 1.5))
        self.drag = (event.x, event.y); self.draw()

    def wheel(self, event):
        self.zoom = float(np.clip(self.zoom * (1.1 if event.delta > 0 else 1/1.1), .4, 4)); self.draw()

    def tick(self):
        try:
            kind, value = self.events.get_nowait(); self.busy = False
            if kind == 'result':
                self.result, self.cfg, self.source, self.motion_cache = value
                self.update_timing_summary()
                if self.motion_cache is not None:
                    key, p, t = self.motion_cache
                    self.motion_caches[key] = (p, t); self.motion_caches.move_to_end(key)
                    while len(self.motion_caches) > 1 and (len(self.motion_caches) > 8 or sum(p.nbytes+t.nbytes for p,t in self.motion_caches.values()) > 128*1024*1024):
                        self.motion_caches.popitem(last=False)
                self.frame = min(self.resume_frame, len(self.result['query_time'])-1)
                self.slider.configure(to=len(self.result['query_time'])-1); self.slider.set(self.frame)
                windows = len(self.result['window_starts'])
                self.status.set(f'{self.source} | 학습 샘플 {windows}개' + (' — 입력 프레임 수를 줄이세요.' if not windows else ''))
                self.draw()
            elif kind == 'archive':
                path, key, members = value
                self.archive_cache[key] = members
                self.select_archive_member(path, members)
            elif kind == 'saved': self.status.set(f'저장 완료: {value}')
            else: messagebox.showerror('Operation failed', value); self.status.set(value)
        except queue.Empty: pass
        if self.playing and self.result is not None:
            now = time.monotonic(); step = int((now-self.last_tick)*self.cfg.output_fps)
            if step:
                self.frame = (self.frame+step) % len(self.result['query_time'])
                self.last_tick += step/self.cfg.output_fps; self.slider.set(self.frame)
        self.root.after(30, self.tick)

    def screen(self, points):
        p = np.asarray(points)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x = p[..., 0]*cy + p[..., 2]*sy
        z = -p[..., 0]*sy + p[..., 2]*cy
        y = p[..., 1]*cp - z*sp
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(w, h)*.29*self.zoom
        return np.stack([w/2+x*scale, h/2-y*scale], -1)

    def line3(self, points, color, width=1, dash=None):
        self.canvas.create_line(*self.screen(points).ravel(), fill=color, width=width, dash=dash)

    def draw_camera(self, origin, rotation, intrinsics, color, dash=None):
        corners = camera_frame(origin, rotation, intrinsics, self.cfg.width, self.cfg.height)
        self.line3(corners[[0, 1, 2, 3, 0]], color, 2, dash)
        for corner in corners:
            self.line3([origin, corner], color, 1, dash)
        # Draw the optical axis plus an up marker so roll errors are visible too.
        self.line3([origin, origin + .65*rotation[2]], color, 2, dash)
        top = (corners[0] + corners[1])/2
        self.line3([top, top - .1*rotation[1]], color, 3, dash)

    def draw(self):
        self.canvas.delete('all'); self.previews.delete('all')
        if self.result is None: return
        d = self.result; i = min(self.frame, len(d['query_time'])-1)
        for axis in range(3):
            for a in (-1, 1):
                for b in (-1, 1):
                    p = np.zeros((2, 3)); p[:, axis] = [-1, 1]
                    other = [j for j in range(3) if j != axis]; p[:, other] = [a, b]
                    self.line3(p, '#334155')
            endpoint = np.eye(3)[axis]*.3
            self.line3([np.zeros(3), endpoint], ['#ef4444', '#22c55e', '#60a5fa'][axis], 2)
            self.canvas.create_text(*self.screen(endpoint), text='XYZ'[axis], fill='white', anchor='sw')
        colors = ['#38bdf8', '#fb923c']
        for hand in range(2):
            p = d['target_xyz'][i, hand]; mask = d['target_mask'][i, hand]
            for finger in range(5):
                chain = [0] + list(range(1+4*finger, 5+4*finger))
                for a, b in zip(chain, chain[1:]):
                    if mask[a] and mask[b]: self.line3(p[[a,b]], colors[hand], 3)
            for xy in self.screen(p[mask]):
                self.canvas.create_oval(xy[0]-3,xy[1]-3,xy[0]+3,xy[1]+3, fill=colors[hand], outline='')
        for c in range(3):
            origin = d['nominal_origins'][c]; actual = d['actual_origins'][c]
            self.draw_camera(origin, d['nominal_rotations'][c], d['nominal_intrinsics'][c], '#94a3b8', (3,5))
            self.draw_camera(actual, d['actual_rotations'][c], d['actual_intrinsics'][c], '#f43f5e')
            nominal_xy = self.screen(origin)
            self.canvas.create_rectangle(nominal_xy[0]-6,nominal_xy[1]-6,nominal_xy[0]+6,nominal_xy[1]+6, outline='#94a3b8')
            xy = self.screen(actual)
            self.canvas.create_oval(xy[0]-3,xy[1]-3,xy[0]+3,xy[1]+3, fill='#f43f5e', outline='')
            self.canvas.create_text(xy[0],xy[1]-16,text=f'C{c+1}',fill='white')
            self.line3([origin, actual], '#f43f5e', 3)
            if self.ray_var.get():
                for h in range(2):
                    if d['input_mask'][i,c,h,0]: self.line3([origin, origin+2*d['ray_directions'][i,c,h,0]], colors[h], dash=(3,3))
            pane_w = self.previews.winfo_width()/3
            s = min((pane_w-18)/self.cfg.width, 145/self.cfg.height)
            ox, oy = c*pane_w+9, 27
            self.previews.create_rectangle(ox,oy,ox+self.cfg.width*s,oy+self.cfg.height*s,outline='#475569')
            count = int(d['input_mask'][i,c].sum())
            self.previews.create_text(ox, 12, text=f'카메라 {c+1} · {count}/42', fill='white', anchor='w')
            if d['selected_frame'][i,c] >= 0:
                capture = d['physical_capture_time'][i,c]
                arrival = d['arrival_time'][i,c]
                delay = (arrival-capture)*1000
                age = (d['query_time'][i]-capture)*1000
                caption = f'촬영→도착 {delay:.1f} ms · 나이 {age:.1f} ms\n촬영 {capture:.3f}s → 도착 {arrival:.3f}s'
            else:
                caption = '사용 가능한 프레임 없음\n미도착 또는 관측 유효 시간 초과'
            self.previews.create_text(ox, 178, text=caption, fill='#cbd5e1', anchor='nw', width=max(100,pane_w-18))
            for h in range(2):
                uv = d['pixels'][i,c,h][d['input_mask'][i,c,h]]*s+[ox,oy]
                for x,y in uv: self.previews.create_oval(x-2,y-2,x+2,y+2,fill=colors[h],outline='')
        now = d['query_time'][i]
        ages = ['—' if d['selected_frame'][i,c]<0 else f'{(now-d["physical_capture_time"][i,c])*1000:.0f}ms' for c in range(3)]
        pose = ' / '.join(f'C{c+1}: {d["actual_position_errors_cm"][c]:.3f} cm · {d["actual_angle_errors_deg"][c]:.3f}°' for c in range(3))
        mode = 'Roll 0° 고정 · Yaw/Pitch 설치 오차는 클립마다 선택됩니다.'
        self.pose_info.set(f'적용된 오차 (클립 동안 고정) — {pose}\n{mode} • 설정 변경 후 적용을 누르세요.')
        self.info.set(f'Frame {i+1}/{len(d["query_time"])} | query {now:.3f}s | age: {" / ".join(ages)} | 1 world = {self.cfg.world_unit_cm:g} cm')


def launch(cfg=None, path=None, member=None):
    root = tk.Tk(); App(root, cfg or Config(), path, member); root.mainloop()


if __name__ == '__main__': launch()
