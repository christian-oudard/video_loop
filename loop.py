import cv2
from imutils.video import WebcamVideoStream
import numpy as np

screen_width = 2560 // 2
screen_height = 1440 // 2


def video_loop(delay):
    """
    Show a delayed video loop with the specified number of seconds of delay buffer.
    """
    vs = WebcamVideoStream().start()

    cv2.namedWindow('loop', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('loop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps = int(vs.stream.get(cv2.CAP_PROP_FPS) * 2)  # Off by a factor of 2, inexplicably.
    width = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf_size = int(delay * fps)
    if buf_size == 0:
        buf_size = 1
    buf = {}
    i = 0
    while True:
        # Capture.
        # Despeckle before main processing.
        buf[i] = vs.read()
        buf[i] = cv2.medianBlur(buf[i], 3)

        # Collect the oldest few frames for frame processing.
        # Start at slow speed until buffer is full and we catch up to it.
        oldest_i = min(buf.keys())
        if i < (2 * buf_size):
            display_i = int(i**2 / (4 * buf_size))
        else:
            display_i = oldest_i
        input_frames = []
        for offset in range(3):
            j = display_i + offset
            if j in buf:
                input_frames.append(buf[j])

        # Display.
        frame = process_frame(input_frames, i)
        cv2.imshow('loop', frame)

        key = cv2.waitKey(1)
        if key != -1:
            # break
            pass

        # Rotate buffer.
        buf.pop(i - buf_size, None)
        i += 1

    cv2.destroyAllWindows()
    vs.stop()


def component_ranges(frame):
    # Axis 0 is the width first, then the height second.
    return np.dstack([
        np.min(np.min(frame, axis=0), axis=0),
        np.max(np.max(frame, axis=0), axis=0),
    ])


def process_frame(frame_list, frame_number):
    frame = average_frames(frame_list)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    frame = np.fliplr(frame)

    # lab_l = frame[:, :, 0]
    # frame[:, :, 0] = lab_l

    factor = screen_height / frame.shape[0]
    frame = cv2.resize(frame, None, fx=factor, fy=factor * 1.01)
    frame = cv2.cvtColor(frame, cv2.COLOR_Lab2BGR)

    return frame


def average_frames(frames):
    # Avoid uint8 overflow.
    avg_frame = np.mean(frames, axis=0)
    avg_frame = avg_frame.astype(np.uint8)
    return avg_frame


def debug_frame(f):
    print(f.dtype, f.shape, f[0, 0])


def brighten_percentile(img, p):
    percentile_brightness = np.percentile(img, p)
    if percentile_brightness == 0:
        brightness_scale = 1
    else:
        brightness_scale = int(255 / percentile_brightness)
    img *= brightness_scale
    np.clip(img, 0, 255, out=img)
    return img


if __name__ == '__main__':
    import sys

    delay_seconds = 2.0
    if len(sys.argv) > 1:
        try:
            delay_seconds = float(sys.argv[1])
        except ValueError:
            pass

    video_loop(delay_seconds)
