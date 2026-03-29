# Live-Webcam-Writing

Энэ төсөл нь вэбкамераас гар дохио зангаа бодит цаг хугацаанд танихын тулд MediaPipe Hands + TensorFlow Lite-г ашигладаг. Энэ нь танд дэлгэцэн дээр долоовор хуруугаараа зураг зурж, нударгаараа дохио зангаагаар арилгах боломжийг олгоно.

## Features

- MediaPipe ашиглан гарны тэмдэглэгээг бодит цагийн горимд илрүүлэх
- TensorFlow Lite загвараар дохио зангааны ангилал
- Виртуал зургийн зураг
- Дохионы хяналтыг ашиглан баллуур горим
- Зүүн болон баруун гарыг дэмжих

## Project Structure

```text
.
├── app.py
├── keypoint_classifier
│   ├── __init__.py
│   ├── keypoint_classifier.py
│   └── keypoint_classifier.tflite
└── training
    ├── dataset.csv
    ├── making_data.ipynb
    └── training_model.ipynb
```

## Requirements

- Python 3.8
- Webcam

Python packages:

- opencv-python=4.13.0.92+
- mediapipe=0.8.1
- numpy=1.24.3+
- tensorflow=2.13.1

## Setup

1. Виртуал орчин үүсгэж, идэвхжүүлэх (санал болгож байна):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Dependencies суулгах:

```bash
pip install opencv-python mediapipe=0.8.10 numpy tensorflow=2.13.1
```

## Run

```bash
python app.py
```

Өөр ажлуулах command arguments:

```bash
python app.py --device 0 --width 960 --height 540 --min_detection_confidence 0.7 --min_tracking_confidence 0.5
```

Useful options:

- `--device`: камерын индекс (defualt: `0`)
- `--width`: зураг авах өргөн (defualt: `960`)
- `--height`: зураг авах өндөр (defualt: `540`)
- `--static_image_mode`: MediaPipe статик зургийн горим
- `--min_detection_confidence`: илрүүлэх итгэлийн босго
- `--min_tracking_confidence`: мөрдөх итгэлийн босго

## Gesture Controls

Сургагдсан загварын таних үйлдлүүд

- Palm
- L
- Fist
- Fist_Moved
- Thumb
- Index
- OK
- Palm_Moved
- C
- Down

Зураг зурах горимд үйлдлүүд зан төлөв:

- `Index`: идэвхтэй хурууны өндгөөр шугам зурах
- `Fist`: хурууны өндгөөр тойруулан арчих
- Бусад дохио зангаа: зурах/арчих үйлдэл хийхгүй

## Training and Data

`training/dataset.csv` дохио зангаа ангилахад ашигладаг урьдчилан боловсруулсан гарын тэмдэглэгээний онцлогуудыг агуулдаг.

Сургалт:

- `training/making_data.ipynb`: өгөгдөл цуглуулах/урьдчилан боловсруулах ажлын урсгал
- `training/training_model.ipynb`: загвар сургалт болон TFLite руу экспортлох

Ажиллах үед ашигласан загвар файл:

- `keypoint_classifier/keypoint_classifier.tflite`

## Notes

- Програмын цонхноос гарахын тулд `Esc` товчийг дарна уу.
