import cv2
import sys
import os

# Correct path to ec_vision/src
ec_vision_src_path = os.path.abspath(
    os.path.join("..", "..", "..", "ec_vision", "src")
)
sys.path.append(ec_vision_src_path)

# Import vision class
from vision import visionNav

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 1 if external cam is on a different port
if not cap.isOpened():
    print("❌ Cannot access camera")
    sys.exit()

nav = visionNav()
print("📸 Starting buoy detection — Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        break

    nav.image = frame
    nav.generate_masks()
    nav.detect_buoys()

    cv2.imshow("Buoy Detection", nav.image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
