import cv2
import tensorflow as tf
import numpy as np
import streamlit as st


RECTANGLE_COLOURS = {
    0: (242, 225, 204),
    1: (213, 247, 251),
    2: (222, 205, 245)
}

POSE_NAMES = [
    "Standing",
    "Sitting",
    "Lying"
]

@st.cache(allow_output_mutation=True)
def load_classifier():
    return tf.lite.Interpreter(model_path="./model/ownlstm.tflite")


CLASSIFIER = load_classifier()
CLASSIFIER.allocate_tensors()

def classifier_prediction_for_person(keypoints_of_person, frame, conf_threshold, coords):
    temp = np.reshape(keypoints_of_person, (1, 51, 1))

    # Setup input and output (Classifier)
    classifier_in = CLASSIFIER.get_input_details()
    classifier_out = CLASSIFIER.get_output_details()

    # Make Classifications
    CLASSIFIER.set_tensor(classifier_in[0]['index'], np.array(temp))
    CLASSIFIER.invoke()
    results = CLASSIFIER.get_tensor(classifier_out[0]['index'])[0]
    classified_pose = np.argmax(results)
    prob = f"{round(max(results)*100, 2)}%"

    if max(results) > conf_threshold:
        draw_classifying_box(frame, coords, classified_pose, prob)


def draw_classifying_box(frame, coords, classified_pose, prob):
    y, x = coords

    # Draw 
    cv2.rectangle(frame,
        (x - 50, y - 30),
        (x + 50, y + 30),
        tuple(RECTANGLE_COLOURS[classified_pose]), -1)

    # Classified pose
    cv2.putText(frame, POSE_NAMES[classified_pose], [x - 50, y - 10],
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    # Probabilities
    cv2.putText(frame, prob, [x - 50, y + 20],
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)


