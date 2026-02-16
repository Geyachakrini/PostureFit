import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time


# ==============================
# Utility Functions
# ==============================

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


# ==============================
# Streamlit UI Setup
# ==============================

st.title("PostureFit - Live Pose Detection")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])


# ==============================
# MediaPipe Setup
# ==============================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==============================
# Session State Initialization
# ==============================

session_defaults = {
    "right_knee_angle_history": [],
    "left_knee_angle_history": [],
    "back_angle_history": [],
    "torso_history": [],
    "rep_count": 0,
    "squat_state": "UP",
    "current_rep_scores": [],
    "rep_scores": [],
    "down_start_time": None,
    "up_start_time": None,
    "last_tempo_feedback": ""
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==============================
# Main Camera Loop
# ==============================

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

            # --------------------------------
            # Preprocessing
            # --------------------------------
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:

                # --------------------------------
                # Landmark Extraction
                # --------------------------------
                landmarks = results.pose_landmarks.landmark
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

                # Right side
                right_shoulder = landmark_array[12][:2]
                right_hip = landmark_array[24][:2]
                right_knee = landmark_array[26][:2]
                right_ankle = landmark_array[28][:2]

                # Left side
                left_shoulder = landmark_array[11][:2]
                left_hip = landmark_array[23][:2]
                left_knee = landmark_array[25][:2]
                left_ankle = landmark_array[27][:2]

                # Midpoints
                mid_shoulder = (right_shoulder + left_shoulder) / 2
                mid_hip = (right_hip + left_hip) / 2
                vertical_point = np.array([mid_hip[0], mid_hip[1] - 0.1])

                # --------------------------------
                # Angle Calculations
                # --------------------------------
                torso_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)
                back_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # --------------------------------
                # History Buffers (Smoothing)
                # --------------------------------
                st.session_state.torso_history.append(torso_angle)
                st.session_state.back_angle_history.append(back_angle)
                st.session_state.right_knee_angle_history.append(right_knee_angle)
                st.session_state.left_knee_angle_history.append(left_knee_angle)

                if len(st.session_state.right_knee_angle_history) > 10:
                    st.session_state.right_knee_angle_history.pop(0)

                if len(st.session_state.left_knee_angle_history) > 10:
                    st.session_state.left_knee_angle_history.pop(0)

                if len(st.session_state.back_angle_history) > 10:
                    st.session_state.back_angle_history.pop(0)

                if len(st.session_state.torso_history) > 10:
                    st.session_state.torso_history.pop(0)

                smoothed_right_knee = np.mean(st.session_state.right_knee_angle_history)
                smoothed_left_knee = np.mean(st.session_state.left_knee_angle_history)
                smoothed_back_angle = np.mean(st.session_state.back_angle_history)
                smoothed_torso = np.mean(st.session_state.torso_history)

                # --------------------------------
                # Symmetry Detection
                # --------------------------------
                knee_difference = abs(smoothed_right_knee - smoothed_left_knee)

                if knee_difference < 10:
                    symmetry_feedback = "Balanced"
                elif knee_difference < 20:
                    symmetry_feedback = "Slight Imbalance"
                else:
                    symmetry_feedback = "Uneven Squat"

                # --------------------------------
                # State Machine + Tempo
                # --------------------------------
                threshold_low = 100
                threshold_high = 160
                current_time = time.time()

                if smoothed_right_knee < threshold_low and st.session_state.squat_state == "UP":
                    st.session_state.squat_state = "DOWN"
                    st.session_state.down_start_time = current_time

                if smoothed_right_knee > threshold_high and st.session_state.squat_state == "DOWN":

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

                # --------------------------------
                # Form Evaluation (DOWN phase only)
                # --------------------------------
                good_depth = smoothed_right_knee < 110
                good_back = smoothed_back_angle > 110
                good_torso = smoothed_torso > 90

                ideal_knee_angle = 80
                ideal_back_angle = 120

                # --------------------------------
                # Display Information
                # --------------------------------
                cv2.putText(image, f"Reps: {st.session_state.rep_count}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 255), 2, cv2.LINE_AA)
                h, w, _ = image.shape 
                knee_pixel = (int(right_knee[0] * w), int(right_knee[1] * h)) 
                cv2.putText( image, str(int(smoothed_right_knee)), knee_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA ) 

                back_pixel = (int(right_hip[0] * w),int(right_hip[1] * h)) 
                cv2.putText(image, str(int(smoothed_back_angle)), back_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2, cv2.LINE_AA)

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

                    cv2.putText(image, feedback,
                                (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, color, 2, cv2.LINE_AA)

                    knee_error = abs(smoothed_right_knee - ideal_knee_angle)
                    back_error = abs(smoothed_back_angle - ideal_back_angle)

                    knee_score = max(0, 100 - knee_error * 2)
                    back_score = max(0, 100 - back_error * 1.5)

                    form_score = int((knee_score + back_score) / 2)
                    st.session_state.current_rep_scores.append(form_score)

                    if st.session_state.rep_scores:
                        last_score = st.session_state.rep_scores[-1]
                    else:
                        last_score = 0

                    cv2.putText(image, f"Score: {last_score}",
                                (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2, cv2.LINE_AA)

                    if st.session_state.rep_scores:
                        overall_avg = int(np.mean(st.session_state.rep_scores))

                    if len(st.session_state.rep_scores) > 1:
                        consistency = int(100 - np.std(st.session_state.rep_scores))
                    else:
                        consistency = 100

                    consistency = max(0, consistency)

                    if len(st.session_state.rep_scores) >= 3:

                        cv2.putText(image, f"Average Score: {overall_avg}",
                                    (30, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

                        cv2.putText(image, f"Consistency: {consistency}",
                                    (30, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, f"Tempo: {st.session_state.last_tempo_feedback}",
                                (30, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, f"Symmetry: {symmetry_feedback}",
                                (30, 290),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (200, 200, 200), 2, cv2.LINE_AA)

                # --------------------------------
                # Draw Landmarks
                # --------------------------------
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(203, 192, 255), thickness=2, circle_radius=3)
                )

            FRAME_WINDOW.image(image)

    cap.release()
