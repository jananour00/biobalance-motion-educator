# BioBalance: Interactive Biomechanics Learning Tool

## 📌 Introduction

Understanding body mechanics during physical activities like squats is essential for **injury prevention**, **performance enhancement**, and **rehabilitation**. A core concept in this domain is the **Center of Mass (COM)**—a point that represents the average location of the body's mass. 

Despite its importance, students and trainees often find it challenging to visualize and analyze COM dynamics during movement. Traditional methods, such as motion capture systems, are often **expensive**, **non-intuitive**, and **not widely accessible**.

![Alt text](https://drive.google.com/uc?export=view&id=1gNL_n0C4Ga2sK7eFDlVuiszBQA3GooHG)
 
## 🎯 Objective

**BioBalance** is an interactive, web-based learning tool designed to teach users—particularly students and trainees—how the **Center of Mass shifts during a squat**. The platform merges:

- Real-time motion tracking using **YOLOv5**
- A biomechanical **Unity-based COM game**
- Interactive quizzes and performance feedback

Together, these elements deliver an engaging educational experience grounded in biomechanics.

---

## ❗ Problem Statement

### Challenges in Biomechanics:

- Accurate analysis of squat motion requires precise joint tracking.
- Traditional equipment is costly and often inaccessible.
- The COM concept is abstract and difficult to visualize.
- Lack of feedback may cause improper squat form, increasing injury risks.

### The Need for a Solution:

- COM tracking improves posture correction and rehabilitation.
- Interactive, visual tools enhance understanding and retention.
- Affordable, web-based solutions improve accessibility for students, athletes, and healthcare professionals.

---

## 💡 Proposed Solution

We developed a **comprehensive web-based platform** that bridges theoretical biomechanics with intuitive and gamified learning.

### 🔧 Key Features:

- **Real-time Squat Tracking:**  
  Powered by **YOLOv5**, the system detects body joints to analyze squat depth and estimate the Center of Mass dynamically.

- **Center of Mass Game (Unity + FLSdk):**  
  A Unity-based game challenges users to estimate the COM based on posture. It reinforces biomechanical reasoning through visual feedback.

- **Interactive Quizzes:**  
  Assess understanding of biomechanics concepts, providing instant feedback to aid learning.

- **Personalized Performance Feedback:**  
  Users receive form suggestions (e.g., shift COM backward) to refine squat execution and posture.

---

## 🛠️ Technologies Used

| Component             | Technology         |
|----------------------|--------------------|
| Motion Tracking       | YOLOv5, Python     |
| Backend Framework     | Flask              |
| Game Engine           | Unity + FLSdk      |
| Web Interface         | HTML, CSS, JS      |
| Visualization Bridge  | Flask API endpoints |

---

## 📈 Results and Findings

### ✅ System Achievements:

- Accurate tracking of squat motion using YOLOv5.
- Real-time COM visualization improved user awareness of proper form.
- Educational game and quizzes increased student engagement.
- Users demonstrated improved understanding of COM after interaction with the system.

### 📊 Visual Data (If Applicable):

- Heatmaps of COM trajectories during squats.
- Quiz performance scores before and after using the tool.
- User feedback reports indicating increased clarity in biomechanical concepts.

---

## 💬 Discussion and Conclusion

### 🧩 Challenges Encountered:

- Lighting conditions affected the accuracy of YOLOv5 detection.
- Latency issues were observed when processing real-time video input.
- Integrating Unity WebGL smoothly into the web interface required custom handling.

### 🚀 Potential Improvements:

- Incorporating full-body multi-joint tracking using MediaPipe or OpenPose.
- Expanding the game with different postures and exercise types.
- Adding a dashboard for performance history and progress tracking.
- Improving UI/UX with accessibility and mobile support.

### 🏁 Conclusion:

**BioBalance** successfully demonstrates how combining **motion tracking**, **game-based learning**, and **interactive feedback** can transform the way students engage with complex biomechanics concepts. It offers a scalable, low-cost alternative to traditional tools and holds promise for further applications in sports science, physical therapy, and education.

---

## 📚 License

MIT License (or specify your preferred license here)

---

## 🙌 Acknowledgments

Thanks to the development team, testers, and academic mentors who guided the project. Special credit to open-source libraries and frameworks like YOLOv5, Unity, and Flask for enabling this interactive solution.

---

## 📬 Contact

For feedback or collaboration inquiries, please contact:

**[Your Name]**  
[Your Email]  
[LinkedIn / GitHub / Website]

