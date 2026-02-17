# PostureFit – Real-Time AI Squat Analyzer

PostureFit is a real-time squat form analysis system built using MediaPipe Pose, OpenCV, and Streamlit.  
It detects body landmarks from a webcam feed, calculates joint angles, counts repetitions, and evaluates squat performance.

---

## 🚀 Features

- Real-time squat detection
- Rep counter using state-machine logic
- Tempo classification
- Gives Feedback
- Form evaluation (depth & back alignment)
- Knee & torso angle display (live on video)
- Symmetry detection
- Moving average smoothing for stability
- Rep scoring & consistency calculation
- Performance summary graphs
- Modern animated UI

---

## 🧠 Core Logic

- Pose landmarks extracted using **MediaPipe**
- Angles computed using vector mathematics
- Rep detection based on knee angle thresholds
- Scoring calculated during DOWN phase
- Consistency = 100 − standard deviation of rep scores

---

## 🛠 Tech Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- NumPy

---

## 📂 Project Structure
PostureFit/
│
├── app.py
├── requirements.txt
└── README.md

---

## ⚙ Installation

```bash
git clone https://github.com/your-username/posturefit.git
cd posturefit
pip install -r requirements.txt
streamlit run app.py

---

## 📈 Performance Metrics Explanation

PostureFit evaluates squat quality using the following computed metrics:

- **Reps**  
  Counted using a state-machine approach (UP → DOWN → UP transition).

- **Tempo**  
  Calculated using time difference between DOWN and UP states:
  - < 0.5s → Too Fast  
  - 0.5–1.2s → Good Tempo  
  - > 1.2s → Slow & Controlled  

- **Form Score (Per Rep)**  
  Computed during the DOWN phase using:
  - Knee angle deviation from ideal (~80°)
  - Back angle deviation from ideal (~120°)  
  Final score = average of knee and back scores.

- **Average Score**  
  Mean of all completed rep scores.

- **Consistency**  
  `100 − standard deviation of rep scores`  
  Higher value = more stable performance.

- **Symmetry**  
  Based on absolute difference between left and right knee angles:
  - < 10° → Balanced  
  - 10–20° → Slight Imbalance  
  - > 20° → Uneven Squat  

All angle values are smoothed using a rolling average buffer to reduce noise.

---

## ⚠ Limitations

- Performance may degrade in poor lighting conditions
- Depends on MediaPipe landmark detection accuracy
- No user calibration mechanism
- No persistent data storage
- Currently supports only squats
- Hardcoded threshold values (not dynamically adaptive)

---

## 🚀 Future Enhancements

- Multi-exercise support (Pushups, Lunges, Deadlifts, etc.)
- Adaptive threshold tuning based on user biomechanics
- Modular code architecture (separate logic, UI, pose modules)
- Real-time live performance graphs
- User profile & progress tracking
- Data export (CSV / JSON)
- Cloud deployment (Streamlit Cloud / Docker)
- AI-based posture correction suggestions
- Angle visualization overlays
- Front-view detection optimization
- Model fine-tuning for higher biomechanical accuracy

---

## 👩‍💻 Author

**Geya Chakrini**  
B.Tech CSE Student  
Computer Vision & AI Enthusiast  

This project was built as a real-time computer vision system to explore biomechanics, pose estimation, and intelligent feedback systems using MediaPipe and OpenCV.

Focused on building practical AI systems that combine logic, UI design, and real-time processing.

---


