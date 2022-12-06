import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st
from MoveNet_Drawing_Utils import draw_skeleton
from MoveNet_Classifier_Utils import classifier_prediction_for_person
import cv2


# Load Models
@st.cache(allow_output_mutation=True)
def load_movenet():
    interpreter = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    return interpreter.signatures['serving_default']



MOVENET = load_movenet()


def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M


def movenet_processing(frame, max_people=6, mn_conf=0.5, kp_conf=0.3, pred_conf=0.5, draw_movenet_skeleton = True):
    height, width = frame.shape[:2]

    
    '''Then run MoveNet and Classifier'''
    # Reshape image for processing
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.int32)

    # MoveNet detections
    res = MOVENET(input_image)
    keypoints_with_scores = res['output_0'].numpy()[:,:,:51].reshape((6,17,3))


    for i in range(len(keypoints_with_scores)):
        if i >= max_people:
            break
        # Note: person is normalised keypoints, but keypoints_with_scores represent the actual coordinates on the frame
        person = keypoints_with_scores[i]

        confidence = sum([person[j][2] for j in range(17)])/17
        if confidence > mn_conf:

            kp_with_scores = person.copy()
            M = get_affine_transform_to_fixed_sizes_with_padding((height, width), (192, 192))
            M = np.vstack((M, [0, 0, 1]))
            M_inv = np.linalg.inv(M)[:2]
            xy_keypoints = kp_with_scores[:, :2] * 192
            xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
            kp_with_scores = np.hstack((xy_keypoints, kp_with_scores[:, 2:]))
            coords = [int(kp_with_scores.flatten()[i]) for i in range(2)] # For classifying box

            # Rendering 
            if draw_movenet_skeleton:
                draw_skeleton(frame, kp_with_scores, kp_conf)
            classifier_prediction_for_person(person, frame, pred_conf, coords)
    return frame

        
