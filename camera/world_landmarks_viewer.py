"""Orbitable 3D wireframes from a local MediaPipe Hand Landmarker camera."""
import argparse
import math
from pathlib import Path
import time

from master import EDGES
from sender import HandTracker, add_camera_arguments


class OrbitView:
    def __init__(self):
        self.reset()
        self.drag = None

    def reset(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0

    def project(self, point, center):
        """Rotate around the model origin, then apply perspective (metres)."""
        x, y, z = point
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, z = cy * x + sy * z, -sy * x + cy * z
        y, z = cp * y - sp * z, sp * y + cp * z
        scale = 1100 * self.zoom * 0.6 / max(0.05, 0.6 + z)
        return (round(center[0] + x * scale), round(center[1] + y * scale)), z

    def mouse(self, event, x, y, flags, cv2):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            if not flags & cv2.EVENT_FLAG_LBUTTON:
                self.drag = None
                return
            self.yaw += (x - self.drag[0]) * 0.01
            self.pitch -= (y - self.drag[1]) * 0.01
            self.drag = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # HighGUI stores a signed wheel delta in the upper 16 bits.
            delta = (flags >> 16) & 0xffff
            if delta >= 0x8000:
                delta -= 0x10000
            self.zoom = min(3.0, max(0.25, self.zoom * (1.12 if delta > 0 else 1 / 1.12)))


def draw_scene(cv2, np, view, hand, title, show_ids):
    panel = np.full((560, 440, 3), (28, 24, 22), dtype=np.uint8)
    center = (220, 290)

    def line(a, b, color, thickness=1):
        cv2.line(panel, view.project(a, center)[0], view.project(b, center)[0],
                 color, thickness, cv2.LINE_AA)

    # Grid spacing is 2 cm, in the model's XY plane; no landmark rescaling.
    for step in range(-7, 8):
        v = step * 0.02
        line((v, -0.14, 0), (v, 0.14, 0), (48, 43, 40))
        line((-0.14, v, 0), (0.14, v, 0), (48, 43, 40))
    for endpoint, label, color in (
        ((0.06, 0, 0), '+X', (100, 100, 255)),
        ((0, 0.06, 0), '+Y', (100, 230, 100)),
        ((0, 0, 0.06), '+Z', (255, 170, 90)),
    ):
        line((0, 0, 0), endpoint, color, 2)
        cv2.putText(panel, label, view.project(endpoint, center)[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if hand:
        projected = [view.project(p, center) for p in hand]
        colors = [(100, 180, 255), (100, 240, 240), (140, 240, 100),
                  (255, 190, 100), (230, 120, 240)]
        # Far segments first gives more natural overlap while orbiting.
        for a, b in sorted(EDGES, key=lambda e: (projected[e[0]][1] + projected[e[1]][1]), reverse=True):
            color = colors[min(4, (b - 1) // 4)]
            cv2.line(panel, projected[a][0], projected[b][0], color, 3, cv2.LINE_AA)
        for i in sorted(range(len(hand)), key=lambda i: projected[i][1], reverse=True):
            xy = projected[i][0]
            cv2.circle(panel, xy, 4, (245, 245, 245), -1, cv2.LINE_AA)
            if show_ids:
                cv2.putText(panel, str(i), (xy[0] + 6, xy[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, 'No hand detected', (130, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(panel, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1)
    cv2.putText(panel, 'Hand-local metres | grid: 2 cm', (16, 540),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.set_defaults(id=1, model=str(Path(__file__).resolve().parent / 'models' / 'hand_landmarker.task'))
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('fps: 1..60')
    if not Path(args.model).is_file():
        parser.error('Model missing: run python camera/download_model.py')
    import cv2
    import numpy as np

    view = OrbitView()
    window = 'MediaPipe World Landmarks - 3D'
    paused, show_ids = False, False
    frame, meta = None, None
    try:
        with HandTracker(args) as tracker:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, lambda e, x, y, f, _: view.mouse(e, x, y, f, cv2))
            print('Drag: orbit | Wheel / +/-: zoom | Space: freeze | R: reset | I: IDs | Q/Esc: exit')
            next_capture = 0.0
            while True:
                now = time.monotonic()
                if frame is None or (not paused and now >= next_capture):
                    frame, meta = tracker.read()
                    next_capture = time.monotonic() + 1 / args.fps
                camera = np.full((560, 360, 3), 22, dtype=np.uint8)
                preview = frame.copy()
                h, w = preview.shape[:2]
                for hand in meta['landmarks']:
                    points = [(round(p[0] * w), round(p[1] * h)) for p in hand]
                    for a, b in EDGES:
                        cv2.line(preview, points[a], points[b], (80, 230, 100), 1, cv2.LINE_AA)
                camera[80:320, 20:340] = preview
                for j, text in enumerate((
                    'PAUSED' if paused else 'LIVE CAMERA',
                    f'Hands: {len(meta["world_landmarks"])} | Infer: {meta["inference_ms"]:.0f} ms',
                    'Drag: rotate    Wheel / +/-: zoom',
                    'Space: freeze   R: reset view',
                    'I: joint IDs    Q / Esc: exit',
                    'Each hand has its own origin.',
                )):
                    y = 30 if j == 0 else 345 + (j - 1) * 34
                    cv2.putText(camera, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (230, 230, 230), 1, cv2.LINE_AA)
                panels = [camera]
                for i in range(args.hands):
                    hand = meta['world_landmarks'][i] if i < len(meta['world_landmarks']) else None
                    name = meta['handedness'][i] if hand else 'Waiting'
                    panels.append(draw_scene(cv2, np, view, hand, f'Hand {i + 1}: {name}', show_ids))
                cv2.imshow(window, np.concatenate(panels, axis=1))
                key = cv2.waitKey(15) & 0xff
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(' '):
                    paused = not paused
                elif key in (ord('r'), ord('R')):
                    view.reset()
                elif key in (ord('i'), ord('I')):
                    show_ids = not show_ids
                elif key in (ord('+'), ord('=')):
                    view.zoom = min(3.0, view.zoom * 1.12)
                elif key == ord('-'):
                    view.zoom = max(0.25, view.zoom / 1.12)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
