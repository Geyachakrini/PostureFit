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

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 40px;'>PostureFit - Live Pose Detection</h1>",
    unsafe_allow_html=True
)






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

left_col, spacer_col, right_col = st.columns([3, 0.3, 2])

with left_col:
    st.markdown("### Video Feed")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

with right_col:
    st.markdown("### Analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    rep_placeholder = st.empty()
    feedback_placeholder = st.empty()
    score_placeholder = st.empty()
    avg_placeholder = st.empty()
    consistency_placeholder = st.empty()
    tempo_placeholder = st.empty()
    symmetry_placeholder = st.empty()

graph_placeholder = st.empty()
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

                # Landmark Extraction
                landmarks = results.pose_landmarks.landmark
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

                right_shoulder = landmark_array[12][:2]
                right_hip = landmark_array[24][:2]
                right_knee = landmark_array[26][:2]
                right_ankle = landmark_array[28][:2]

                left_shoulder = landmark_array[11][:2]
                left_hip = landmark_array[23][:2]
                left_knee = landmark_array[25][:2]
                left_ankle = landmark_array[27][:2]

                mid_shoulder = (right_shoulder + left_shoulder) / 2
                mid_hip = (right_hip + left_hip) / 2
                vertical_point = np.array([mid_hip[0], mid_hip[1] - 0.1])

                # Angle Calculations
                torso_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)
                back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # Smoothing Buffers
                st.session_state.torso_history.append(torso_angle)
                st.session_state.back_angle_history.append(back_angle)
                st.session_state.right_knee_angle_history.append(right_knee_angle)
                st.session_state.left_knee_angle_history.append(left_knee_angle)

                for key in [
                    "torso_history",
                    "back_angle_history",
                    "right_knee_angle_history",
                    "left_knee_angle_history"
                ]:
                    if len(st.session_state[key]) > 10:
                        st.session_state[key].pop(0)

                smoothed_right_knee = np.mean(st.session_state.right_knee_angle_history)
                smoothed_left_knee = np.mean(st.session_state.left_knee_angle_history)
                smoothed_back_angle = np.mean(st.session_state.back_angle_history)
                smoothed_torso = np.mean(st.session_state.torso_history)

                # Symmetry
                knee_difference = abs(smoothed_right_knee - smoothed_left_knee)

                if knee_difference < 10:
                    symmetry_feedback = "Balanced"
                elif knee_difference < 20:
                    symmetry_feedback = "Slight Imbalance"
                else:
                    symmetry_feedback = "Uneven Squat"

                # Safe metric defaults
                if st.session_state.rep_scores:
                    last_score = st.session_state.rep_scores[-1]
                    overall_avg = int(np.mean(st.session_state.rep_scores))
                else:
                    last_score = 0
                    overall_avg = 0

                if len(st.session_state.rep_scores) > 1:
                    consistency = int(100 - np.std(st.session_state.rep_scores))
                else:
                    consistency = 100

                consistency = max(0, consistency)

                # State Machine
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

                # Form Evaluation
                feedback = "—"

                if st.session_state.squat_state == "DOWN":
                    good_depth = smoothed_right_knee < 110
                    good_back = smoothed_back_angle > 110

                    if smoothed_right_knee < 140:
                        if good_depth and good_back:
                            feedback = "Good Form"
                        elif not good_back:
                            feedback = "Straighten Your Back"
                        else:
                            feedback = "Go Lower"

                    knee_error = abs(smoothed_right_knee - 80)
                    back_error = abs(smoothed_back_angle - 120)

                    knee_score = max(0, 100 - knee_error * 2)
                    back_score = max(0, 100 - back_error * 1.5)

                    form_score = int((knee_score + back_score) / 2)
                    st.session_state.current_rep_scores.append(form_score)

                # Update Metrics
                rep_placeholder.markdown(f"**Reps:** {st.session_state.rep_count}")
                feedback_placeholder.markdown(f"**Feedback:** {feedback}")
                score_placeholder.markdown(f"**Last Rep Score:** {last_score}")
                tempo_placeholder.markdown(f"**Tempo:** {st.session_state.last_tempo_feedback}")
                avg_placeholder.markdown(f"**Average Score:** {overall_avg}")
                consistency_placeholder.markdown(f"**Consistency:** {consistency}")
                symmetry_placeholder.markdown(f"**Symmetry:** {symmetry_feedback}")

                # Draw Landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(203, 192, 255), thickness=2, circle_radius=3)
                )

            FRAME_WINDOW.image(image)

    cap.release()


st.markdown("---")
st.markdown("### Performance Graphs")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.markdown("#### Avg Score per Rep")
    if st.session_state.rep_scores:
        st.line_chart(st.session_state.rep_scores)
    else:
        st.write("No reps yet")

with graph_col2:
    st.markdown("#### Last Score vs Rep")
    if st.session_state.rep_scores:
        rep_numbers = list(range(1, len(st.session_state.rep_scores)+1))
        bar_data = {
            "Rep": rep_numbers,
            "Score": st.session_state.rep_scores
        }
        st.bar_chart(bar_data)
    else:
        st.write("No reps yet")
