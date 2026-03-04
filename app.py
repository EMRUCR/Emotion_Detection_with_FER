import cv2
from fer import FER
import time
import numpy as np


canvas = np.zeros((720,1280,3), dtype=np.uint8) + 120

detector = FER(mtcnn=False)

cap = cv2.VideoCapture(0)
secondwindow = cv2.imread("neutral.jpg")

alpha = 0.8
smoothed = None
prev = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.detect_emotions(rgb)

    for r in results:
        # 1) Choosing only one Face 
        r = max(results, key=lambda x: x["box"][2] * x["box"][3])
        x, y, w, h = r["box"]
        emotions = r["emotions"]  # dict

        # 2) Smoothing (EMA)
        if smoothed is None:
            smoothed = emotions.copy()
        else:
            for k in emotions.keys():
                smoothed[k] = alpha * smoothed[k] + (1 - alpha) * emotions[k]

        # En yüksek duygu (smoothed üzerinden)
        emotion = max(smoothed, key=smoothed.get)
        score = smoothed[emotion]

        if (emotion == "happy"):
            secondwindow = cv2.imread("happy.jpg")
        
        elif (emotion == "sad"):
            secondwindow = cv2.imread("sad.jpg")

        elif (emotion == "angry"):
            secondwindow = cv2.imread("angry.jpg")

        elif (emotion == "surprise"):
            secondwindow = cv2.imread("surprised.jpg")

        elif (emotion == "fear"):
            secondwindow = cv2.imread("fear.jpg")

        else:
            secondwindow = cv2.imread("neutral.jpg")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev) if prev else 0
    prev = now
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    frame = cv2.resize(frame, (560,360))
    canvas[50:410,50:610] = frame

    secondwindow = cv2.resize(secondwindow, (560,360))
    canvas[50:410, 670:1230] = secondwindow

    cv2.namedWindow("FER Emotion (Press q)", cv2.WINDOW_NORMAL)
    cv2.imshow("FER Emotion (Press q)", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()