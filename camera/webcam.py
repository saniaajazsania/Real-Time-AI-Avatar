import cv2


def start_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Webcam started successfully.")

    return cap