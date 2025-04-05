# 🎥 Video Analytics Project

A real-time video analytics system built using **OpenCV**, **TensorFlow**, and **Flask** that performs human and product identification, recognition, and availability detection.

---

## 🔍 Features

### 1. 👤 Human Identification
- Utilizes **OpenCV** for real-time video capture.
- Employs **TensorFlow** models for identifying human presence.
- Integrated with **Flask** for a simple web-based interface.

### 2. 🧍‍♂️🧱 Human and Product Recognition
- Detects and distinguishes between **humans** and **products** in a video stream.
- Color-coded bounding boxes:
  - **Green** box for humans.
  - **Blue** box for products.

### 3. ✅📦 Product Availability Check
- System checks if the detected product is available.
- Displays:
  - **"Product is Available"** if detected in stock.
  - **"Product is Not Available"** if not detected or out of stock.

---

## 🛠 Tech Stack

- **Python**
- **OpenCV** for image/video processing
- **TensorFlow/Keras** for model inference
- **Flask** for serving the web interface
- Optional: **DeepFace**, **NumPy**, **Pandas**, etc.

---

## 🚀 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/video-analytics-project.git
   cd video-analytics-project
   
2.**Create and activate virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate




