# Face Recognition System 🎯

This is a Python-based face recognition project using the `face_recognition` and `OpenCV` libraries. It compares known faces in a dataset with test images and identifies matches.

## 🔍 Features

- Detects and recognizes human faces in images.
- Uses a tolerance value for flexible accuracy.
- Visualizes matches using bounding boxes and labels.
- Configurable model (`hog` or `cnn`) for detection.

## 🗂️ Project Structure

```
├── dataset/
│   ├── Person1/
│   │   └── image1.jpg
│   └── Person2/
│       └── image2.jpg
├── Test Data/
│   └── test1.jpg
├── face_recognition_main.py
└── README.md
```

## 🧠 Requirements

- Python 3.x
- face_recognition
- OpenCV (`cv2`)
- numpy

Install them via pip:
```bash
pip install face_recognition opencv-python numpy
```

## 🚀 How to Run

1. Place your known face images in the `dataset/PersonName/` folders.
2. Place test images in the `Test Data/` folder.
3. Run the script:
```bash
python face_recognition_main.py
```

4. The script will open windows showing each test image with matched names and bounding boxes.

## ⚙️ Configuration

You can modify these parameters in the script:

```python
tolerance = 0.6         # Lower is stricter (0.4–0.6 recommended)
MODEL = "cnn"           # Use "hog" for faster but less accurate detection
```
## 📁 Notes

- Make sure images are clear and faces are well-lit.
- The CNN model needs a GPU or will run slowly on CPU.

## 📝 Credits
- Tutorial : sentdex/youtube
- Dataset : Kaggle
