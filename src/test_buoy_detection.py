import sys
import os
import cv2

# Add path to ec_vision/src manually
ec_vision_path = r"D:\IAAC\Term 2\Engineering club\ec_vision\src"
sys.path.append(ec_vision_path)

from vision import visionNav


# Initialize camera
cap = cv2.VideoCapture(0)  # Try 1 if 0 doesn't work

if not cap.isOpened():
    print("❌ Cannot access camera")
    sys.exit()

nav = visionNav()

print("📸 Starting buoy detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        break

    nav.image = frame
    nav.generate_masks()
    nav.detect_buoys()

    # Show result
    cv2.imshow("Buoy Detection", nav.image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
