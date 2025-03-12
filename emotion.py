import cv2
from deepface import DeepFace as df

# Load Haar Cascade for face detection
# Ensure the XML file path is correct; replace with the full path if needed
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the webcam (camera index 0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Main loop to process video feed
while cap.isOpened():
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If frame not captured successfully, exit the loop
    if not ret:
        break

    # Convert the frame to grayscale for better face detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    # scaleFactor: Parameter specifying the image size reduction at each image scale
    # minNeighbors: Specifies how many neighbors each rectangle should have to retain it
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate through all detected faces
    for x, y, w, h in faces:
        # Draw a red rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the detected face region from the frame
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Analyze the cropped face for emotions using DeepFace
            # actions=['emotion']: Focuses on detecting the dominant emotion
            analyze = df.analyze(face_roi, actions=['emotion'])

            # Extract the dominant emotion from the analysis result
            emotion = analyze[0]['dominant_emotion']
            print(f"Emotion: {emotion}")

            # Display the detected emotion above the face rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 55, 50), 2)

        except Exception as e:
            # Handle cases where emotion analysis fails
            print("Error during emotion analysis or no face detected:")

    # Show the video feed with detected faces and emotions in a window
    cv2.imshow('video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam resource and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
