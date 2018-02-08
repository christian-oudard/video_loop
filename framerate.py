import time
import cv2
import math


def main():
    times = time_frames()
    diffs = [ b - a for (a, b) in pairwise(times) ]
    print('framerate: {:.2f} fps'.format(1.0 / average(diffs)))
    print('frame length deviation: {:.5f}s'.format(stddev(diffs)))


def time_frames(sample_size=100):
    cap = cv2.VideoCapture(0)  # 0 = default camera

    times = []
    times.append(time.time())

    for _ in range(sample_size):
        ret, img = cap.read()
        # cv2.imshow('img', img)  # This slows down the framerate, use threading.
        times.append(time.time())

    cap.release()

    return times


def average(seq):
    return sum(seq) / len(seq)


def stddev(seq):
    avg = average(seq)
    variance = sum( (x - avg)**2 for x in seq ) / len(seq)
    return math.sqrt(variance)


def pairwise(seq):
    iseq = iter(seq)
    iseq2 = iter(seq)
    next(iseq2)
    return zip(iseq, iseq2)

if __name__ == '__main__':
    main()
