from heading_reader import get_heading

heading = get_heading("/dev/ttyACM0")  # Update if needed
if heading is not None:
    print(f"🧭 Current Heading: {heading:.2f}°")
