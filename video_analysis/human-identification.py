import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the camera
cap = cv2.VideoCapture(0)

print("[INFO] Starting human detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8))

    # Draw bounding boxes around detected people
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Human Detection", frame_resized)

    # Exit loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
