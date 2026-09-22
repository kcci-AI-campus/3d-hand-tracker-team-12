"""Display Hand Landmarker results from this device's camera without networking."""
import argparse
from pathlib import Path
import time

from master import EDGES
from sender import HandTracker, add_camera_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.set_defaults(
        id=1,
        model=str(Path(__file__).resolve().parent / 'models' / 'hand_landmarker.task'))
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('fps: 1..60')
    if not Path(args.model).is_file():
        parser.error('Model missing: run python download_model.py')

    import cv2

    window = 'Local Hand Landmarker'
    try:
        with HandTracker(args) as tracker:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            print('Local camera ready. Press Q, Esc, or Ctrl+C to exit.')
            while True:
                start = time.monotonic()
                frame, meta = tracker.read()
                panel = frame.copy()
                height, width = panel.shape[:2]
                for hand in meta['landmarks']:
                    points = [
                        (max(0, min(width - 1, round(p[0] * width))),
                         max(0, min(height - 1, round(p[1] * height))))
                        for p in hand]
                    for a, b in EDGES:
                        cv2.line(panel, points[a], points[b], (0, 255, 0), 1)
                    for point in points:
                        cv2.circle(panel, point, 2, (0, 0, 255), -1)
                label = (f'Hands: {len(meta["landmarks"])}  '
                         f'Infer: {meta["inference_ms"]:.0f}ms')
                cv2.rectangle(panel, (0, 0), (width - 1, 24), (0, 0, 0), -1)
                cv2.putText(panel, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)
                cv2.imshow(window, panel)
                delay_ms = max(1, round((1 / args.fps - (time.monotonic() - start)) * 1000))
                if cv2.waitKey(delay_ms) & 0xFF in (27, ord('q'), ord('Q')):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
