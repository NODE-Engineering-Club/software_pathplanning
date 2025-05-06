#sudo apt install -y libcamera-dev python3-picamera2
#pip install opencv-python


import sys, os
from picamera2 import Picamera2
import cv2
import time

# Add ec_vision/src to Python path
ec_vision_src = os.path.abspath(os.path.join("..", "ec_vision", "src"))
sys.path.append(ec_vision_src)

from vision import visionNav

# Initialize Raspberry Pi camera
picam = Picamera2()
picam.preview_configuration.main.size = (640, 480)
picam.preview_configuration.main.format = "RGB888"
picam.configure("preview")
picam.start()

time.sleep(1)  # Warm-up time

# Initialize vision system
nav = visionNav()
print("📸 Starting buoy detection — Press ESC to stop.")

while True:
    try:
        frame = picam.capture_array()

        # Run buoy detection
        nav.image = frame.copy()
        nav.generate_masks()
        nav.detect_buoys()

        # Add instructions on frame
        cv2.putText(nav.image, "Press ESC to stop", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show output
        cv2.imshow("Buoy Detection", nav.image)

        # Exit when ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("❌ Error:", e)
        break

# Cleanup
picam.close()
cv2.destroyAllWindows()
