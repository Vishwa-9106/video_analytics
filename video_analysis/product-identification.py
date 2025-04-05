import cv2
from ultralytics import YOLO

# Load YOLO model (YOLOv5 or YOLOv8)
model = YOLO('yolov5s.pt')  # pre-trained on COCO dataset

# Define product class you want to detect (e.g., bottles = class 39)
PRODUCT_CLASS_ID = 39  # Bottle
MIN_COUNT_THRESHOLD = 1  # Set to 0 for 'no product' alert

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Product availability monitoring started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    product_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # If detected class is our target product
        if cls_id == PRODUCT_CLASS_ID:
            product_count += 1
            label = f'Product ({conf:.2f})'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Status label
    if product_count <= MIN_COUNT_THRESHOLD:
        status = f"⚠ Product NOT available!"
        color = (0, 0, 255)
    else:
        status = f"✅ Product Available: {product_count}"
        color = (0, 255, 0)

    cv2.putText(frame, status, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Product Availability Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
