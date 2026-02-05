# Face-Recognition-Based-Smart-Door-Lock-System
This is a **student project** developed using **Python**.  
The project is based on **face recognition technology** to allow or deny access
to a door automatically.

The system recognizes authorized faces using a webcam and unlocks the door
with the help of **Arduino and an electric lock / servo motor**.

---

## 📌 Project Objective

The main objective of this project is:
- To learn **Python programming**
- To understand **face recognition concepts**
- To implement a **real-time security system**
- To integrate **software with hardware (Arduino)**

---

## ⚙️ Technologies Used

- Python
- OpenCV
- face_recognition library
- NumPy
- Arduino
- Webcam

---

## ✨ Features

- Face detection using webcam
- Face recognition for known users
- Door unlock for authorized person
- Door remains locked for unknown person
- Voice message for access granted / denied
- Saves image of unknown person
- Simple and easy to use

---

## 📁 Project Files

.
├── app.py # Main Python program
├── dataset/ # Stored images of known persons
├── intruders/ # Images of unknown persons
├── README.md # Project documentation


---

## 🛠️ Requirements

Install the required libraries using pip:

```bash
pip install opencv-python face-recognition numpy pyttsx3 pyserial
▶️ How to Run the Project
Connect webcam to the computer

Connect Arduino and door lock system

Run the program:

python app.py
Show your face in front of the camera

If face is recognized → door unlocks

If face is unknown → access denied

🔒 Applications
Home security system

Office access control

College or lab entry system

Learning purpose for students

⚠️ Limitations
Works best in good lighting

Limited number of users

Not suitable for large-scale systems

📚 Learning Outcome
Through this project, I learned:

Python programming basics

Face recognition concepts

Using OpenCV in real projects

Hardware and software integration

👨‍🎓 Developed By
Abu Huraira
Student of Artificial Intelligence
