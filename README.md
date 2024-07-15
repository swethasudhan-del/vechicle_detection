# vechicle_detection
we have designed a simple project to detect the vechicle movements
pip install opencv-python
import cv2

# Load the recorded video
cap = cv2.VideoCapture('traffic_video.mp4')  # Replace with your video file name

# Load the vehicle detection model
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 1)  # Detect vehicles

    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangles around vehicles
        cv2.putText(frame, 'Cut-in Detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print("Alert: Vehicle cut-in detected!")  # Print alert for simplicity

    cv2.imshow('Vehicle Detection', frame)  # Show the frame in a window
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()  # Release the video file
cv2.destroyAllWindows()  # Close the window
python cut_in_detection_from_video.py
pip install playsound
import cv2
from playsound import playsound
import threading

def play_alert():
    playsound('alert_sound.mp3')

# Load the recorded video
cap = cv2.VideoCapture('traffic_video.mp4')  # Replace with your video file name

# Load the vehicle detection model
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 1)  # Detect vehicles

    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangles around vehicles
        cv2.putText(frame, 'Cut-in Detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print("Alert: Vehicle cut-in detected!")
        threading.Thread(target=play_alert).start()  # Play sound alert

    cv2.imshow('Vehicle Detection', frame)  # Show the frame in a window
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()  # Release the video file
cv2.destroyAllWindows()  # Close the window
