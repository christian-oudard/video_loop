# TODO
# Threading for capture.

import cv2
import numpy as np

def video_loop(delay):
    """
    Show a delayed video loop with the specified number of seconds of delay buffer.
    """
    cv2.namedWindow('loop', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('loop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)  # 0 = default camera

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf_size = int(delay * fps)
    buf = np.zeros(
        (buf_size, height, width, 3),
        dtype=np.uint8,
    )
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.fliplr(frame)

        cv2.imshow('loop', buf[i])
        buf[i] = frame

        i = (i + 1) % buf_size

        if cv2.waitKey(1) != -1:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_loop(1.5)
