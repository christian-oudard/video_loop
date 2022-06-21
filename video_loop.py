import time

import cv2
import imutils
from imutils.video import WebcamVideoStream
import numpy as np

screen_width = 2560 // 2
screen_height = 1440 // 2

FPS = 120


def video_loop(delay):
    """
    Show a delayed video loop with the specified number of seconds of delay buffer.
    """
    vs = WebcamVideoStream().start()

    cv2.namedWindow('loop', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('loop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    buf_size = 500
    if buf_size == 0:
        buf_size = 1
    buf = [None] * buf_size
    buf_index = 0

    timestamps = []

    while True:
        # timestamps.append(time.monotonic())
        try:
            pass
            # print(1 / (timestamps[-1] - timestamps[-11]))
            # print('fps: {}'.format(round(1/10 * 1 / timestamps[-1] - timestamps[-11])))
        except IndexError:
            pass

        frame = vs.read()
        frame = np.fliplr(frame)

        buf[buf_index] = frame
        buf_index += 1
        buf_index %= buf_size

        oldest_frame = buf[(buf_index + 1) % buf_size]
        if oldest_frame is not None:
            cv2.imshow('loop', oldest_frame)

        key = cv2.waitKey(1)
        if key != -1:
            if key == 27:  # Esc.
                return

        end_ts = time.monotonic()


    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    import sys

    delay_seconds = 1.0
    if len(sys.argv) > 1:
        try:
            delay_seconds = float(sys.argv[1])
        except ValueError:
            pass

    video_loop(delay_seconds)
