"""PC side: receive the master Pi's H3D1 packets and render both hands in 3D in real time.

    pip install numpy pyqtgraph PyQt6 PyOpenGL
    python hand_viewer.py                    # UDP 6000
    python hand_viewer.py --port 6000 --smooth .5 --hide-unseen

Hands: bones as thick lines, joints as dots, the palm as a translucent surface; left hand blue,
right hand orange. A hand that is not there (HandLiteV3: in-view probability below 0.5; HandDirect:
no camera saw it recently, so its joints are the network's guess) is drawn faint (or hidden with
--hide-unseen). The three cameras of models/rig.json are drawn as small
pyramids looking at the origin, with a 10 cm floor grid. Drag to orbit, wheel to zoom.
Packet format: pc_receiver.py / src/pc_packet.hpp."""
import argparse
import json
from pathlib import Path
import socket
import sys
import threading
import time
import numpy as np

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    sys.exit('Needs pyqtgraph with Qt and OpenGL:  pip install pyqtgraph PyQt6 PyOpenGL')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc_receiver import BONES, decode, present  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
COLORS = ((.25, .55, 1.), (1., .55, .15))          # left, right
PALM = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), (1, 2, 5)]   # wrist fan over the knuckles
FINGER_BONES = np.array(BONES).reshape(-1)          # segment endpoints for GL 'lines' mode


def to_view(points_cm):
    """Rig frame (Y up) -> the viewer's frame (Z up), a rotation: (x, y, z) -> (x, -z, y)."""
    p = np.asarray(points_cm, np.float32)
    return np.stack((p[..., 0], -p[..., 2], p[..., 1]), -1)


class Receiver(threading.Thread):
    """Keeps the newest packet and packet statistics; runs in the background."""

    def __init__(self, port, swap_hands=False):
        super().__init__(daemon=True)
        self.swap_hands = swap_hands
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(.2)
        self.lock = threading.Lock()
        self.latest, self.received, self.bad, self.times = None, 0, 0, []
        self.running = True

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            packet = decode(data, self.swap_hands)
            with self.lock:
                if packet is None:
                    self.bad += 1
                    continue
                self.latest = packet
                self.received += 1
                now = time.monotonic()
                self.times = [t for t in self.times if now-t < 1.]+[now]

    def snapshot(self):
        with self.lock:
            return self.latest, self.received, self.bad, len(self.times)


class HandItem:
    """One hand's bones, joints and palm surface."""

    def __init__(self, view, color):
        self.color = color
        self.bones = gl.GLLinePlotItem(mode='lines', width=6, antialias=True)
        self.joints = gl.GLScatterPlotItem(size=11, pxMode=True)
        faces = np.array(PALM, np.int32)
        self.palm = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=np.zeros((21, 3), np.float32), faces=faces),
                                  smooth=False, drawEdges=False, shader=None, glOptions='translucent')
        self.faces = faces
        for item in (self.palm, self.bones, self.joints):
            view.addItem(item)

    def set(self, joints_cm, alpha):
        if alpha <= 0:
            for item in (self.bones, self.joints, self.palm):
                item.setVisible(False)
            return
        p = to_view(joints_cm)
        r, g, b = self.color
        segments = p[FINGER_BONES]
        self.bones.setData(pos=segments, color=(r, g, b, alpha))
        # Fingertips a little lighter, wrist darker, so the pose reads at a glance.
        shade = np.ones((21, 4), np.float32)*(r, g, b, alpha)
        shade[[4, 8, 12, 16, 20], :3] = np.clip(np.array((r, g, b))*1.3+.1, 0, 1)
        shade[0, :3] *= .6
        self.joints.setData(pos=p, color=shade)
        self.palm.setMeshData(vertexes=p, faces=self.faces, faceColors=np.tile((r, g, b, .35*alpha), (len(self.faces), 1)))
        for item in (self.bones, self.joints, self.palm):
            item.setVisible(True)


def camera_items(view, rig_path):
    """Small pyramids at the rig's cameras, pointing along their view directions."""
    try:
        rig = json.loads(Path(rig_path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    unit = rig['world_unit_cm']
    for c, (origin, rotation) in enumerate(zip(rig['origins'], rig['rotations'])):
        r = np.asarray(rotation, float)                       # rows: right, down, forward (world axes)
        o = np.asarray(origin, float)*unit
        size = 6.
        corners = [o+size*(r[2]*1.5+sx*r[0]*.8+sy*r[1]*.6) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        lines = []
        for k in range(4):
            lines += [o, corners[k], corners[k], corners[(k+1) % 4]]
        lines += [corners[0], corners[0]+(corners[1]-corners[0])*.5-r[1]*3]   # "up" tick on the image top
        item = gl.GLLinePlotItem(pos=to_view(np.array(lines)), mode='lines', width=2, color=(.8, .8, .8, .9), antialias=True)
        view.addItem(item)
        label = gl.GLTextItem(pos=to_view(o-r[1]*4), text=f'cam{c}', color=(220, 220, 220, 255))
        view.addItem(label)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=6000)
    p.add_argument('--rig', default=str(HERE/'models'/'rig.json'), help='Draw these cameras (optional)')
    p.add_argument('--smooth', type=float, default=0., help='Exponential smoothing 0..0.95 of the drawn joints (0: raw)')
    p.add_argument('--hide-unseen', action='store_true', help='Hide a hand no camera saw instead of drawing it faint')
    p.add_argument('--swap-hands', action='store_true', help='Exchange the left and right hand when reading')
    args = p.parse_args()
    if not 0 <= args.smooth < 1:
        p.error('--smooth must be in [0, 1)')

    receiver = Receiver(args.port, args.swap_hands)
    receiver.start()
    app = QtWidgets.QApplication(sys.argv)
    view = gl.GLViewWidget()
    view.setWindowTitle(f'Hand tracker 3D — UDP {args.port}')
    view.setBackgroundColor((22, 24, 28))
    view.setCameraPosition(distance=140, elevation=18, azimuth=-90)
    view.resize(1100, 800)
    grid = gl.GLGridItem()
    grid.setSize(120, 120)
    grid.setSpacing(10, 10)
    grid.translate(0, 0, -40)                                # the floor 40 cm below the rig origin
    view.addItem(grid)
    axes = gl.GLAxisItem()
    axes.setSize(10, 10, 10)
    view.addItem(axes)
    camera_items(view, args.rig)
    hands = [HandItem(view, color) for color in COLORS]
    status = QtWidgets.QLabel(view)
    status.setStyleSheet('color: #ddd; background: rgba(0,0,0,110); padding: 6px; font: 12px monospace;')
    status.move(10, 10)
    view.show()

    drawn = [None, None]
    last_sequence = [None]

    def update():
        packet, received, bad, rate = receiver.snapshot()
        if packet is None:
            status.setText(f'Waiting for H3D1 packets on UDP {args.port} …')
            status.adjustSize()
            return
        if packet['sequence'] != last_sequence[0]:
            last_sequence[0] = packet['sequence']
            for h in range(2):
                joints = packet['joints_cm'][h]
                drawn[h] = joints if drawn[h] is None or not args.smooth else args.smooth*drawn[h]+(1-args.smooth)*joints
                alpha = 1. if present(packet, h) else (0. if args.hide_unseen else .25)
                hands[h].set(drawn[h], alpha)
        view = '' if packet['in_view'] is None else f"  in-view L/R {packet['in_view'][0]:.2f}/{packet['in_view'][1]:.2f}"
        text = (f"{rate} fps  latency {packet['latency_ms']:.0f} ms  seen L/R {packet['seen'][0]}/{packet['seen'][1]}{view}\n"
                f"#{packet['sequence']}  packets {received}  bad {bad}")
        status.setText(text)
        status.adjustSize()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)                                           # ~60 Hz redraw; packets arrive at ~12 Hz
    try:
        app.exec()
    finally:
        receiver.running = False


if __name__ == '__main__':
    main()
