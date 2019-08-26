import cv2
import imutils
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
        # buf[i] = cv2.medianBlur(buf[i], 3)

        # Collect the oldest few frames for frame processing.
        # Start at slow speed until buffer is full and we catch up to it.
        oldest_i = min(buf.keys())
        if i < (2 * buf_size):
            display_i = int(i**2 / (4 * buf_size))
        else:
            display_i = oldest_i

        # input_frames = []
        # for offset in range(3):
        #     j = display_i + offset
        #     if j in buf:
        #         input_frames.append(buf[j])

        # Display.
        frame = process_frame(buf[oldest_i], i)
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


def process_frame(frame, frame_number):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
    # k_images = kaleidoscope_images(frame, frame_number, n=13, flip=False, speed=0)
    k_images = kaleidoscope_images(frame, frame_number, n=7, flip=True, speed=0.2)
    # k_images = kaleidoscope_images(frame, frame_number, n=3, flip=False, speed=0)
    # k_images = kaleidoscope_images(frame, frame_number, n=2, flip=True, speed=0)
    # k_images = kaleidoscope_images(frame, frame_number, n=1, flip=True, speed=0)
    frame = cv2.addWeighted(maximum_blend(k_images), 0.7, minimum_blend(k_images), 0.3, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_XYZ2BGR)

    frame = center_frame(frame)
    frame = trim_circle(frame)
    return frame


def trim_circle(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) // 2
    blur_size = 7

    alpha = np.zeros((height, width, 3), np.uint8)
    cv2.circle(alpha, (center_x, center_y), radius - (blur_size // 2), (255, 255, 255), thickness=-1)
    alpha = cv2.blur(alpha, (blur_size, blur_size))

    alpha = alpha.astype(float) / 255
    frame = frame.astype(float)
    out = cv2.multiply(alpha, frame) / 255
    return out


def kaleidoscope_images(frame, frame_number, n=3, flip=True, speed=0.5):
    images = []
    base_angle = speed * frame_number
    for i in range(n):
        angle = base_angle + (360 / n) * i
        rotated = imutils.rotate(frame, angle)
        images.append(rotated)
        if flip:
            images.append(np.fliplr(rotated))
    return images


def maximum_blend(images):
    result = images[0]
    for img in images[1:]:
        result = cv2.max(result, img)
    return result


def minimum_blend(images):
    result = images[0]
    for img in images[1:]:
        result = cv2.min(result, img)
    return result


def center_frame(frame):
    frame = imutils.resize(frame, height=screen_height, inter=cv2.INTER_LINEAR)
    bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    left = screen_width // 2 - frame.shape[1] // 2
    right = left + frame.shape[1]
    bg[:, left:right] = frame
    return bg


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


def component_ranges(frame):
    # Axis 0 is the width first, then the height second.
    return np.dstack([
        np.min(np.min(frame, axis=0), axis=0),
        np.max(np.max(frame, axis=0), axis=0),
    ])


if __name__ == '__main__':
    import sys

    delay_seconds = 2.0
    if len(sys.argv) > 1:
        try:
            delay_seconds = float(sys.argv[1])
        except ValueError:
            pass

    video_loop(delay_seconds)
