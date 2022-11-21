import cv2
import numpy as np
import matplotlib
import numpy as np

EDGES = {
    (0, 1)  : (191, 0, 191),
    (0, 2)  : (0, 191, 191),
    (1, 3)  : (191, 0, 191),
    (2, 4)  : (0, 191, 191),
    (0, 5)  : (191, 0, 191),
    (0, 6)  : (0, 191, 191),
    (5, 7)  : (191, 0, 191),
    (7, 9)  : (191, 0, 191),
    (6, 8)  : (0, 191, 191),
    (8, 10) : (0, 191, 191),
    (5, 6)  : (191, 191, 0),
    (5, 11) : (191, 0, 191),
    (6, 12) : (0, 191, 191),
    (11, 12): (191, 191, 0),
    (11, 13): (191, 0, 191),
    (13, 15): (191, 0, 191),
    (12, 14): (0, 191, 191),
    (14, 16): (0, 191, 191)
}


def draw_skeleton(frame, keypoints_with_scores, confidence_threshold):
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)
    draw_connections(frame, keypoints_with_scores, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [1,1,1]))
    # print(shaped)
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,0,0), -1) 

def draw_connections(frame, keypoints, confidence_threshold):

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [1,1,1]))
    
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)