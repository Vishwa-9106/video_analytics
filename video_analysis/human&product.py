import cv2
from ultralytics import YOLO

# Load YOLOv5 model (requires 'ultralytics' library)
model = YOLO('yolov5s.pt')  # You can also use yolov8n.pt or yolov5m.pt

# Class names of COCO dataset used by YOLO
# 0 = person, other IDs = product-like classes
product_ids = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # e.g., bottles, cups, laptops...

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Detecting humans and products... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 0:  # Person
            color = (0, 255, 0)  # Green
            label = f'Person {conf:.2f}'
        elif cls_id in product_ids:
            color = (255, 0, 0)  # Blue
            label = f'{model.names[cls_id]} {conf:.2f}'
        else:
            continue  # Skip other objects

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Human and Product Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
