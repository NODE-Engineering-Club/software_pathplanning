#sudo apt install -y libcamera-dev python3-picamera2
#pip install opencv-python

import sys, os
from picamera2 import Picamera2
import cv2
import time

# Add ec_vision/src to the path (adjust path depth if needed)
ec_vision_src = os.path.abspath(os.path.join("..", "ec_vision", "src"))
sys.path.append(ec_vision_src)

from vision import visionNav  # Import your vision logic

# Initialize Pi Camera
picam = Picamera2()
picam.preview_configuration.main.size = (640, 480)
picam.preview_configuration.main.format = "RGB888"
picam.configure("preview")
picam.start()

time.sleep(1)  # Allow camera to warm up

nav = visionNav()  # Setup buoy detection logic
print("📸 Buoy detection running... Press ESC to quit.")

while True:
    try:
        frame = picam.capture_array()

        nav.image = frame.copy()
        nav.generate_masks()
        nav.detect_buoys()

        # Overlay decision (if available)
        decision_text = f"Nav: {nav.direction}" if hasattr(nav, "direction") else ""
        cv2.putText(nav.image, decision_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(nav.image, "Press ESC to stop", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Buoy Detection - PiCam", nav.image)

        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        break

picam.close()
cv2.destroyAllWindows()
