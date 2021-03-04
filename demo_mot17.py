from gmphd import *
import os
from os import path
import collections
import numpy as np
import cv2
import time
import multiprocessing as mp


def read_mot(relpath='./MOT17-02/'):
    names = collections.defaultdict(list)
    detections = collections.defaultdict(list)

    # Store the image names.
    for file in os.listdir(path.join(relpath, 'img1')):
        if file.endswith('.jpg'):
            name, extension = file.split('.')
            names[int(name)] = file

    # Load the detections.
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>
    with open(path.join(relpath, 'gt/gt.txt'), mode='r') as file:
        for line in file:
            line = line.replace('\n', '')
            frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y = map(float, line.split(','))
            detections[frame].append(np.array([bb_left, bb_top, bb_width, bb_height]).reshape(4, 1))

    return names, detections


if __name__ == '__main__':
    # state [x y dx dy].T constant velocity model
    F = np.array([[1, 0, 1, 0],  # state transition matrix
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    P = np.array([[5 ** 2, 0, 0, 0],  # covariance matrix of state
                  [0, 10 ** 2, 0, 0],
                  [0, 0, 5 ** 2, 0],
                  [0, 0, 0, 10 ** 2]])
    Q = np.array([[5 ** 2, 0, 0, 0],  # process noise covariance
                  [0, 10 ** 2, 0, 0],
                  [0, 0, 5 ** 2, 0],
                  [0, 0, 0, 10 ** 2]]) * 1 / 2
    H = np.array([[1, 0, 0, 0],  # observation matrix
                  [0, 1, 0, 0]])
    R = np.array([[5 ** 2, 0],  # observation noise covariance
                  [0, 10 ** 2]])
    pdf_c = 2.5e-07  # clutter intensity

    im_width = 1545
    im_height = 1080
    birthprob = 0.1  # 0.05 # 0 # 0.2
    survivalprob = 0.9  # 0.95 # 1
    detectprob = 0.99  # 0.999
    bias = 1  # 8   # tendency to prefer false-positives over false-negatives in the filtered output
    birthgmm = []
    # Note: I have noticed that the birth gmm needs to be narrow/fine,
    # because otherwise it can lead the pruning algo to lump foreign components together
    for x in range(0, im_width, 200):
        for y in range(0, im_height, 200):
            state = np.array([x, y, 0, 0])
            gmphd = GmphdComponent(weight=1e-3, loc=state, cov=P)
            birthgmm.append(gmphd)
    print('Ended Initial GmphdComponent')

    tracker = Gmphd(birthgmm, survivalprob, detection=detectprob, f=F, q=Q, h=H, r=R, clutter=pdf_c)
    names, detections = read_mot()
    pool = mp.Pool(processes=mp.cpu_count())

    for frame in range(min(names.keys()), max(names.keys())):
        # Perform a prediction-update step.
        start = time.time()
        obs = numpy.array(detections[frame], dtype=float)
        tracker.update_mp(obs[:, :2] + (obs[:, 2:] / 2.0), pool)  # center of bbox
        tracker.prune(truncthresh=1e-3, mergethresh=5, maxcomponents=len(obs) + 50)
        fps = time.time() - start

        integral = sum(np.array([comp.weight for comp in tracker.gmm]))
        estitems = tracker.extractstatesusingintegral(bias=bias)

        image = cv2.imread(path.join('./MOT17-02/img1', names[frame]))
        for comp in estitems:
            (x, y), id = (comp[0][0], comp[0][1]), comp[1]
            image = cv2.circle(image, (x, y), radius=8,
                               color=(255, 255, 255), thickness=-1)
            image = cv2.putText(image, str(id),
                                org=(x+5, y+5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                                color=(0, 255, 255), thickness=2)

        # Plot the detections.
        #if frame in detections:
        #   for x, y, u, v in detections[frame]:
        #        image = cv2.rectangle(image, (x, y), (x + u, y + v), color=(0, 0, 0), thickness=2)
        image = cv2.putText(image, 'Frame {}'.format(frame) + ', FPS:{}'.format(round(1 / fps, 2)),
                            org=(im_width - 400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2)
        cv2.imwrite('./MOT17-02/output/' + str(frame) + '.jpg', image)
        cv2.imshow('Image', image)
        cv2.waitKey(1)

    # making video from output images
    os.system("ffmpeg -r 30 -i ./MOT17-02/output/%d.jpg -vcodec mpeg4 -y ./MOT17-02/MOT17-02.mp4")
