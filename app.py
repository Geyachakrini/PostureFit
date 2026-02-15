import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle=np.clip(cosine_angle,-1.0,1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle
st.title("PostureFit - Live Pose Detection")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if "angle_history" not in st.session_state:
    st.session_state.angle_history = []

if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0

if "squat_state" not in st.session_state:
    st.session_state.squat_state = "UP"


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
                landmarks = results.pose_landmarks.landmark
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

                right_shoulder = landmark_array[12][:2]
                right_hip = landmark_array[24][:2]
                right_knee = landmark_array[26][:2]
                right_ankle = landmark_array[28][:2]

                back_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                st.session_state.angle_history.append(knee_angle)

                # Keep only last 10 values
                if len(st.session_state.angle_history) > 10:
                    st.session_state.angle_history.pop(0)
                # Smoothed angle
                smoothed_angle = np.mean(st.session_state.angle_history)
                # Define Thresholds
                threshold_low = 90
                threshold_high = 160
                # State Machine Logic
                if smoothed_angle < threshold_low:
                    st.session_state.squat_state = "DOWN"

                if smoothed_angle > threshold_high and st.session_state.squat_state == "DOWN":
                    st.session_state.squat_state = "UP"
                    st.session_state.rep_count += 1
                
                cv2.putText(image,
                            f"Reps: {st.session_state.rep_count}",
                            (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,255),
                            2,
                            cv2.LINE_AA)
                
                h, w, _ = image.shape
                knee_pixel = (int(right_knee[0] * w), int(right_knee[1] * h))

                cv2.putText(
                        image,
                        str(int(smoothed_angle)),
                        knee_pixel,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA
                        )
                
                good_depth = smoothed_angle < 100
                good_back = back_angle > 150

                if st.session_state.squat_state == "DOWN":
                    if good_depth and good_back:
                        feedback = "Good Form"
                        color = (0, 255, 0)
                    elif not good_back:
                        feedback = "Straighten Your Back"
                        color = (255, 223, 0)
                    else:
                        feedback = "Go Lower"
                        color = (255, 223, 0)

                    cv2.putText(
                        image,
                        feedback,
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA)  
                      
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,0,255),thickness=2,circle_radius=3),
                    mp_drawing.DrawingSpec(color=(203, 192, 255),thickness=2,circle_radius=3)
                )

            FRAME_WINDOW.image(image)

    cap.release()


