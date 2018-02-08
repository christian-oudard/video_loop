# TODO
# Threading for capture.

import cv2
from imutils.video import WebcamVideoStream
import numpy as np


def video_loop(delay):
    """
    Show a delayed video loop with the specified number of seconds of delay buffer.
    """
    vs = WebcamVideoStream().start()

    cv2.namedWindow('loop', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('loop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps = int(vs.stream.get(cv2.CAP_PROP_FPS))
    width = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf_size = int(delay * fps)
    buf = np.zeros(
        # (buf_size, height, width),
        (buf_size, height, width, 3),
        dtype=np.uint8,
    )
    i = 0
    while True:
        # Capture.
        frame = vs.read()

        frame = np.fliplr(frame)
        buf[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        i = (i + 1) % buf_size

        # Display.
        n_frames = 3
        frames = [ buf[(i + j) % buf_size] for j in range(n_frames) ]
        frame = sum( f // n_frames for f in frames )

        lab_l = frame[:, :, 0]

        blurred = cv2.medianBlur(lab_l, 5)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        brighten_percentile(laplacian, 99.9)

        float_lab_l = np.float64(lab_l) - laplacian
        np.clip(float_lab_l, 0, 255, out=float_lab_l)
        lab_l = np.uint8(float_lab_l)

        frame[:, :, 0] = lab_l

        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2BGR)

        cv2.imshow('loop', frame)

        key = cv2.waitKey(1)
        if key != -1:
            # break
            pass

    cv2.destroyAllWindows()
    vs.stop()


def brighten_percentile(img, p):
    percentile_brightness = np.percentile(img, p)
    if percentile_brightness == 0:
        brightness_scale = 1
    else:
        brightness_scale = int(255 / percentile_brightness)
    img *= brightness_scale


if __name__ == '__main__':
    video_loop(3)
