import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

# Open the camera
video_capture = cv2.VideoCapture(0)  # 0 represents the default camera

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crop the detected face
        face_roi = frame[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_detector.detectMultiScale(face_roi)

        index = 0
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            if index == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif index == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)

            cv2.rectangle(face_roi, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 255, 255), 2)
            index = index + 1

        # Check if both eyes were found
        if 'eye_1' in locals() and 'eye_2' in locals():
            # Perform further operations with eye_1 and eye_2
            # For example, you can calculate the distance between the eyes
            distance_between_eyes = abs(eye_1[0] - eye_2[0])
            print("Distance between eyes:", distance_between_eyes)
        else:
            print("Both eyes not found.")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
