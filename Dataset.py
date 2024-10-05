import cv2
import numpy as np
import os
import pickle

# Initialize video capture
video = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not video.isOpened():
    print("Error: Could not open video stream or file.")
    exit()

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_data = []

i = 0
name = input("Enter Your Name: ")

while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Failed to capture image. Exiting.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Only save face data for every 10th frame and if under 100 captures
        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
        # Show the current frame
        cv2.imshow("Frame", frame)
        
    i += 1
    k = cv2.waitKey(1)

    # Stop when 50 face images are captured
    if len(face_data) == 25:
        break

# Release the video capture object
video.release()
cv2.destroyAllWindows()

# Reshape the collected face data
face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

# Check if 'data' directory exists, create if it doesn't
if not os.path.exists('data/'):
    os.makedirs('data/')

# Save the captured face data
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Handle the faces data correctly
if 'faces_data.pkl' not in os.listdir('data/'):
    # If faces_data.pkl doesn't exist, create it
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    # Load the existing faces data
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    # Check if 'faces' is empty before attempting concatenation
    if faces.size == 0:
        faces = face_data
    else:
        faces = np.append(faces, face_data, axis=0)
    
    # Save the updated faces data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print("Face data collection complete!")
