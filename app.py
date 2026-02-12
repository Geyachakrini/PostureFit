import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("PostureFit - Live Pose Detection")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Camera not working")
                break

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improve performance-Make only Readable
            image.flags.writeable = False

            # Process image
            results = pose.process(image)

            # Draw landmarks
            # Make it Writable again
            image.flags.writeable = True
            

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,0,255),thickness=2,circle_radius=4),
                    mp_drawing.DrawingSpec(color=(203, 192, 255),thickness=2,circle_radius=4)
                )

            FRAME_WINDOW.image(image)

    cap.release()

