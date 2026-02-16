import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
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

if "back_angle_history" not in st.session_state:
    st.session_state.back_angle_history=[]

if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0

if "squat_state" not in st.session_state:
    st.session_state.squat_state = "UP"

if "current_rep_scores" not in st.session_state:
    st.session_state.current_rep_scores = []

if "rep_scores" not in st.session_state:
    st.session_state.rep_scores = []

if "down_start_time" not in st.session_state:
    st.session_state.down_start_time = None

if "up_start_time" not in st.session_state:
    st.session_state.up_start_time = None

if "last_tempo_feedback" not in st.session_state:
    st.session_state.last_tempo_feedback = ""

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
                st.session_state.back_angle_history.append(back_angle)

                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                st.session_state.angle_history.append(knee_angle)

                # Keep only last 10 values
                if len(st.session_state.angle_history) > 10:
                    st.session_state.angle_history.pop(0)
                # Smoothed angle
                smoothed_angle = np.mean(st.session_state.angle_history)
                # Define Thresholds
                threshold_low = 100
                threshold_high = 160
                # State Machine Logic

                current_time = time.time()

                if smoothed_angle < threshold_low and st.session_state.squat_state == "UP":
                    st.session_state.squat_state = "DOWN"
                    st.session_state.down_start_time = current_time

                if smoothed_angle > threshold_high and st.session_state.squat_state == "DOWN":
                    st.session_state.squat_state = "UP"
                    st.session_state.rep_count += 1

                    up_time = current_time - st.session_state.down_start_time
                    if up_time < 0.5:
                        tempo_feedback = "Too Fast"
                    elif up_time < 1.2:
                        tempo_feedback = "Good Tempo"
                    else:
                        tempo_feedback = "Slow & Controlled"

                    st.session_state.last_tempo_feedback = tempo_feedback

                    if st.session_state.current_rep_scores:
                        avg_score = int(np.mean(st.session_state.current_rep_scores))
                        st.session_state.rep_scores.append(avg_score)

                    st.session_state.current_rep_scores = []

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
                if len(st.session_state.back_angle_history) > 10:
                    st.session_state.back_angle_history.pop(0)
                # Smoothed angle
                
                smoothed_back_angle = np.mean(st.session_state.back_angle_history)
                
                back_pixel = (int(right_hip[0] * w),int(right_hip[1] * h))
                cv2.putText(image,
                            str(int(smoothed_back_angle)),
                            back_pixel,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255,0,255),
                            2,
                            cv2.LINE_AA)
                
                good_depth = smoothed_angle < 110
                good_back = smoothed_back_angle > 110

                ideal_knee_angle = 80
                ideal_back_angle = 120

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
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA)  
                
                    knee_error = abs(smoothed_angle - ideal_knee_angle)
                    back_error = abs(smoothed_back_angle - ideal_back_angle)

                    knee_score = max(0, 100 - knee_error * 2)
                    back_score = max(0, 100 - back_error * 1.5)

                    form_score = int((knee_score + back_score) / 2)
                    st.session_state.current_rep_scores.append(form_score)

                    if st.session_state.rep_scores:
                        last_score = st.session_state.rep_scores[-1]
                    else:
                        last_score=0
                    cv2.putText(
                        image,
                        f"Score: {last_score}",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA)
                    
                    if st.session_state.rep_scores:
                        overall_avg = int(np.mean(st.session_state.rep_scores))
                    
                    if len(st.session_state.rep_scores) > 1:
                        consistency = int(100 - np.std(st.session_state.rep_scores))
                    else:
                        consistency = 100
                    
                    consistency = max(0, consistency)

                    if len(st.session_state.rep_scores) >= 3:
                        cv2.putText(
                                image,
                                f"Avgerage Score: {overall_avg}",
                                (30, 170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA)

                        cv2.putText(
                            image,
                            f"Consistency: {consistency}",
                            (30, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA)

                    cv2.putText(
                            image,
                            f"Tempo: {st.session_state.last_tempo_feedback}",
                            (30, 250),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
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


