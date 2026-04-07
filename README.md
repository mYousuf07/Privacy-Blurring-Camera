# 🔒 Privacy Camera - Real-Time Privacy Blurring Webcam

**Project Exhibition - II**  
*A privacy-first webcam application that automatically pixelates faces and blacks out text in real-time.*

<image-card alt="Privacy Camera Logo" src="assets/logo.svg" ></image-card>

---

## 📋 Project Overview

**Privacy Camera** is a lightweight, floating webcam application that protects your privacy by:

- Applying **heavy, deblur-resistant pixelation** to all detected faces (using elliptical mask for natural look)
- **Completely blacking out** any detected text (documents, screens, IDs, etc.)
- Supporting **trusted faces** — your own face (or team members) can stay visible while others are blurred
- Using a **frame buffer** to ensure **no raw/unprocessed frame ever reaches the screen**

Perfect for video calls, live streaming, online classes, or any situation where you want maximum privacy.

---

## ✨ Key Features

- Advanced Face Anonymization with randomized heavy pixelation
- Real-time Text Detection & Black-box masking using DBNet
- Trusted Faces System (whitelist your face)
- Floating Draggable UI with dark modern theme
- Minimize to logo bubble
- Live statistics (faces blurred, text regions, trusted count)
- Easy camera switching
- Privacy-safe pipeline (black fallback on error)
- GPU acceleration support

---

## 👥 Team Members (Project Exhibition - II)

1. **Prabuddhiraj** - Lead Developer & UI/Backend Integration  
2. **[Member 2 Name]** - Model Integration & Testing  
3. **[Member 3 Name]** - Anonymization Logic & Privacy Features  
4. **[Member 4 Name]** - Text Detection & Optimization  
5. **[Member 5 Name]** - Documentation & Exhibition Presentation  

*(Please replace the placeholders with your actual team members' names and roles)*

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/privacy-camera.git
cd privacy-camera
2. Install Dependencies
Bashpip install -r requirements.txt
Note: onnxruntime-gpu requires CUDA and cuDNN installed on your system.
For CPU-only version, replace onnxruntime-gpu with onnxruntime in requirements.txt.
3. Download Required AI Models (Important!)
⚠️ Models are NOT included in this repository because of their large size.

Create a folder named models in the project root.
Download these two files and place them inside the models/ folder:File NamePurposeApproximate Sizemodel.ptYOLOv8 Face Detection Model~50 MBdbnet_en.onnxDBNet Text Detection Model~2.4 MBHow to get the models:
Ask any team member (models were shared via Google Drive / WhatsApp / college server)
Or download from the link provided during the exhibition

(Optional) Add logo and font:
Place assets/logo.svg and assets/fonts/Roboto-Regular.ttf



🚀 How to Run the Application

Double-click start_app.bat (recommended on Windows)
Or run from terminal:Bashpython main.py

The app will appear as a floating window. Use the "Apply" button to switch between cameras.

📁 Project Structure
textprivacy-camera/
├── main.py                    # Main PyQt6 GUI application
├── anonymization.py           # Face pixelation logic
├── text_detector.py           # Text detection with DBNet
├── start_app.bat              # Windows launcher script
├── requirements.txt
├── trusted_faces.json         # Auto-generated trusted faces list
├── models/                    # ← Put model.pt and dbnet_en.onnx here
├── assets/                    # Fonts and logo (optional)
└── test.py                    # Test ONNX + CUDA setup

🧪 Testing Your Setup
Run the following command to check if models and GPU are working:
Bashpython test.py
You should see CUDAExecutionProvider listed and a success message.

🎯 How to Use

💾 Save My Face — Click when your face is clearly visible to whitelist it
👁 Show Trusted — Toggle to keep trusted faces visible
🗑 — Clear all trusted faces
Drag the window to move it
Double-click the minimized logo to restore the full UI
Switch cameras using the dropdown + Apply button


🛡️ Privacy Guarantees

No raw camera frame is ever displayed
All processing happens before rendering
Black frame shown on any error (no data leak)
Completely offline — no internet or cloud required


🧪 Technologies Used

Python 3
PyQt6 (Floating UI)
Ultralytics YOLOv8 (Face Detection)
DBNet / PaddleOCR ONNX (Text Detection)
OpenCV + NumPy
ONNX Runtime GPU


📄 License
This project is developed for Project Exhibition - II only.
All rights reserved by the development team.

Made with ❤️ for Digital Privacy
Project Exhibition - II | [Your College / Department Name]
