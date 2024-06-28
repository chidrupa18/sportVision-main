import cv2
import numpy as np
import matplotlib.pyplot as plt
from openpose import pyopenpose as op

# Initialize OpenPose
params = {
    "model_folder": "path_to_openpose/models/",  # Path to OpenPose models folder
    "net_resolution": "-1x480",
    "model_pose": "BODY_25",
    "number_people_max": 1
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Initialize Kalman filter parameters
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Initialize video capture
cap = cv2.VideoCapture('your_video.mp4')  # Replace 'your_video.mp4' with the path to your video

# Variables for dribble counting
prev_position = np.array([[0], [0]], dtype=np.float32)  # Initialize with a dummy value
dribble_count = 0
is_ball_upward = False  # Flag to track upward motion

# Lists to store ball positions for plotting
ball_positions_x = []
ball_positions_y = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Get keypoints of the first person detected (assuming only one person is in the frame)
    keypoints = datum.poseKeypoints[0]

    # Get position of nose (or any other keypoint)
    nose_x = int(keypoints[0][0])
    nose_y = int(keypoints[0][1])

    # Predict the next state using the Kalman filter
    prediction = kalman.predict()

    # Correct the state based on the measured values
    measurement = np.array([[nose_x], [nose_y]], dtype=np.float32)
    kalman.correct(measurement)

    # Draw the predicted and corrected positions on the frame
    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(measurement[0]), int(measurement[1])), 5, (0, 255, 0), -1)

    # Store ball positions for plotting
    ball_positions_x.append(measurement[0][0])
    ball_positions_y.append(measurement[1][0])

    # Display dribble count on the video
    cv2.putText(frame, f'Dribbles: {dribble_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Tracking and Counting Dribbles', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Plot ball positions
plt.figure()
plt.plot(ball_positions_x, ball_positions_y, label='Ball Position')
plt.title('Ball Position Over Time')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()
