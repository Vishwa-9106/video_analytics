import cv2
from deepface import DeepFace

# Load webcam
cap = cv2.VideoCapture(0)

print("[INFO] Emotion analysis started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        dominant_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][dominant_emotion]

        label = f"{dominant_emotion} ({confidence:.2f}%)"

        # Draw label
        cv2.putText(frame, label, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if dominant_emotion == 'happy' else (0, 0, 255), 2)

        if dominant_emotion == 'happy':
            cv2.putText(frame, "Customer is HAPPY! ✅", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, "Face not detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Customer Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
