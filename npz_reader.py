"""Read-only NPZ browser with GigaHands input/ground-truth comparison."""
import argparse
import json
import zipfile
import numpy as np


def inventory(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if not entry.filename.endswith('.npy'): continue
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers: raise ValueError(f'Unsupported NPY version: {version}')
                shape, _, dtype = readers[version](stream)
            rows.append((entry.filename[:-4], shape, str(dtype)))
    if not rows: raise ValueError('NPZ에 배열이 없습니다.')
    return rows


def parse_slice(text):
    result = []
    if not text.strip(): return ()
    for item in text.split(','):
        if ':' not in item:
            result.append(int(item)); continue
        values = [int(v.strip()) if v.strip() else None for v in item.split(':')]
        if len(values) > 3 or (len(values) == 3 and values[2] == 0):
            raise ValueError('start:stop:step 형식이며 step은 0일 수 없습니다.')
        result.append(slice(*values))
    return tuple(result)


def compare_sample(data, frame, camera, hand, joint):
    from gigahands_sim import FEATURES
    if not 0 <= frame < data['features'].shape[0]: raise ValueError('프레임 범위를 확인하세요.')
    if not (0 <= camera < 3 and 0 <= hand < 2 and 0 <= joint < 21):
        raise ValueError('카메라 0~2, 손 0~1, 관절 0~20을 입력하세요.')
    x = data['features'][frame,camera,hand,joint]
    y = data['target_xyz'][frame,hand,joint]
    valid = bool(data['input_mask'][frame,camera,hand,joint])
    gt_valid = bool(data['target_mask'][frame,hand,joint])
    lines = [f'프레임 {frame} / 카메라 {camera} / 손 {hand} / 관절 {joint}',
             f'예측·정답 시점: {data["query_time"][frame]:.6f} s',
             f'입력 유효: {valid} / 정답 유효: {gt_valid}', '', 'INPUT — features']
    lines.extend(f'[{i:2}] {name:26} = {float(value): .6f}' for i,(name,value) in enumerate(zip(FEATURES,x)))
    lines.extend(['', 'GROUND TRUTH — target_xyz', f'x = {y[0]: .6f}\ny = {y[1]: .6f}\nz = {y[2]: .6f}',
                  '', '정답은 촬영 시점이 아닌 예측 시점의 월드 좌표입니다.'])
    if 'metadata' in data:
        unit = json.loads(str(data['metadata'])).get('units', {}).get('world_unit_cm')
        if unit is not None:
            lines.append(f'1 world = {unit:g} cm / 정답(cm): {np.array2string(y * unit, precision=3)}')
    for key, label in [('actual_position_errors_cm', '카메라 위치 오차(cm)'), ('actual_angle_errors_deg', '카메라 회전 오차(도)')]:
        if key in data: lines.append(f'{label}: {data[key][camera]:.4f} (진단용, 모델 입력 제외)')
    if not valid:
        lines.append('입력은 결측입니다. 0으로 채운 값을 관측으로 해석하지 마세요.')
        if 'selected_frame' in data and data['selected_frame'][frame,camera] < 0:
            lines.append('원인: 도착한 프레임이 없거나 관측 유효 시간이 지났습니다.')
        elif 'in_frame_mask' in data:
            cause = '추가 랜덤 관절 누락' if data['in_frame_mask'][frame,camera,hand,joint] else '투영된 점이 영상 범위 밖/카메라 뒤 또는 원본 결측'
            lines.append(f'원인: {cause}')
    if not gt_valid: lines.append('정답은 결측입니다. 학습 loss에서 제외하세요.')
    if 'selected_frame' in data and data['selected_frame'][frame,camera] >= 0:
        lines.extend(['', 'TIMESTAMPS — 초'])
        for key in ('capture_time','arrival_time','physical_capture_time'):
            if key in data: lines.append(f'{key}: {data[key][frame,camera]:.6f}')
        lines.append('도착−보고 촬영 시간에는 처리 시간 및 시계 오차도 포함될 수 있습니다.')
    return '\n'.join(lines)


class Reader:
    def __init__(self, root, path=None):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        self.root = root; self.data = None; self.array = None; self.key = None
        root.title('NPZ Dataset Reader'); root.geometry('1200x800'); root.minsize(900,600)
        bar = ttk.Frame(root,padding=8); bar.pack(fill='x')
        ttk.Button(bar,text='Open NPZ',command=self.choose).pack(side='left')
        self.file_label = tk.StringVar(value='읽기 전용 · 원본 파일은 변경하지 않습니다.')
        ttk.Label(bar,textvariable=self.file_label).pack(side='left',padx=12)
        pane = ttk.Panedwindow(root,orient='horizontal'); pane.pack(fill='both',expand=True)
        left = ttk.Frame(pane); pane.add(left,weight=1)
        self.tree = ttk.Treeview(left,columns=('shape','dtype'),show='tree headings',selectmode='browse')
        for key,label,width in [('#0','배열 이름',185),('shape','Shape',145),('dtype','자료형',80)]:
            self.tree.heading(key,text=label); self.tree.column(key,width=width)
        scrollbar = ttk.Scrollbar(left,command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y'); self.tree.pack(fill='both',expand=True)
        self.tree.bind('<<TreeviewSelect>>',self.select)
        right = ttk.Frame(pane,padding=8); pane.add(right,weight=2)
        ttk.Label(right,text='슬라이스: 30,0,0,0 또는 0:3,0,0 / 빈칸: 전체 배열').pack(anchor='w')
        row = ttk.Frame(right); row.pack(fill='x',pady=6)
        self.slice_var = tk.StringVar()
        entry = ttk.Entry(row,textvariable=self.slice_var); entry.pack(side='left',fill='x',expand=True)
        entry.bind('<Return>',lambda e:self.preview())
        ttk.Button(row,text='View Values',command=self.preview).pack(side='left',padx=5)
        group = ttk.LabelFrame(right,text='GigaHands input ↔ ground truth (인덱스는 0부터)',padding=8)
        group.pack(fill='x',pady=6); self.indices = []
        for label,maximum in [('프레임',0),('카메라',2),('손',1),('관절',20)]:
            ttk.Label(group,text=label).pack(side='left',padx=3)
            var = tk.StringVar(value='0'); self.indices.append(var)
            spin = ttk.Spinbox(group,from_=0,to=maximum,textvariable=var,width=5); spin.pack(side='left')
            if label == '프레임': self.frame_spin = spin
        self.compare_button = ttk.Button(right,text='Compare with Ground Truth',command=self.compare,state='disabled')
        self.compare_button.pack(anchor='w',pady=4)
        self.output = ScrolledText(right,wrap='word',font=('Consolas',11)); self.output.pack(fill='both',expand=True)
        self.write('NPZ 파일을 선택하세요.\n배열 목록은 헤더만 읽습니다. 값 조회는 선택한 배열 전체를 메모리에 읽습니다.\nmetadata를 선택하면 생성 설정을 확인할 수 있습니다.')
        root.protocol('WM_DELETE_WINDOW',self.close)
        if path: self.open(path)

    def write(self,text):
        self.output.configure(state='normal'); self.output.delete('1.0','end')
        self.output.insert('1.0',text); self.output.configure(state='disabled')

    def choose(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[('NumPy archive','*.npz')])
        if path: self.open(path)

    def open(self,path):
        from tkinter import messagebox
        from pathlib import Path
        try:
            rows = inventory(path); data = np.load(path,allow_pickle=False)
        except Exception as exc:
            messagebox.showerror('파일 열기 실패',str(exc)); return
        if self.data is not None: self.data.close()
        self.data = data; self.array = None; self.key = None
        self.tree.delete(*self.tree.get_children())
        for key,shape,dtype in rows: self.tree.insert('','end',text=key,values=(str(shape),dtype))
        shapes = {key:shape for key,shape,_ in rows}
        required = {'features','input_mask','target_xyz','target_mask','query_time'}
        supported = required <= shapes.keys() and len(shapes['features']) == 5 and shapes['features'][1:] == (3,2,21,14)
        self.compare_button.configure(state='normal' if supported else 'disabled')
        if supported: self.frame_spin.configure(to=max(0,shapes['features'][0]-1))
        for var in self.indices: var.set('0')
        self.file_label.set(f'{Path(path).name} · {len(rows)}개 배열')
        self.write(f'{Path(path).resolve()}\n\n배열을 클릭하거나 프레임을 선택해 관측과 정답을 비교하세요.\n압축 NPZ의 슬라이스 조회도 해당 배열 전체를 메모리에 읽습니다.')

    def select(self,event=None):
        selected = self.tree.selection()
        if not selected: return
        self.key = self.tree.item(selected[0],'text'); self.array = None
        self.slice_var.set(''); self.preview()

    def preview(self):
        if self.data is None or self.key is None: return
        try:
            if self.array is None: self.array = self.data[self.key]
            a = self.array[parse_slice(self.slice_var.get())]
            header = f'{self.key}[{self.slice_var.get()}]\nshape: {a.shape}\ndtype: {a.dtype}\n\n'
            if a.ndim == 0 and a.dtype.kind in 'US':
                text = str(a.item())
                try: text = json.dumps(json.loads(text),indent=2,ensure_ascii=False)
                except ValueError: pass
                text = text[:50000]
            else:
                text = np.array2string(a,threshold=300,edgeitems=3,precision=6,max_line_width=100)
                text += '\n\n큰 배열은 …로 생략됩니다. 슬라이스를 좁혀 확인하세요.'
            self.write(header+text)
        except Exception as exc: self.write(f'배열을 표시할 수 없습니다: {exc}')

    def compare(self):
        try: self.write(compare_sample(self.data,*(int(v.get()) for v in self.indices)))
        except Exception as exc: self.write(f'비교할 수 없습니다: {exc}')

    def close(self):
        if self.data is not None: self.data.close()
        self.root.destroy()


def launch(path=None):
    import tkinter as tk
    root = tk.Tk(); Reader(root,path); root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?')
    parser.add_argument('--summary',action='store_true')
    args = parser.parse_args()
    if args.summary:
        if not args.path: parser.error('--summary requires a file path')
        for key,shape,dtype in inventory(args.path): print(f'{key:26} {str(shape):24} {dtype}')
    else: launch(args.path)


if __name__ == '__main__': main()
