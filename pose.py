import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

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
cap = cv2.VideoCapture('test.mp4')  # Replace 'your_video.mp4' with the path to your video

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Variables for dribble counting
prev_position = np.array([[0], [0]], dtype=np.float32)  # Initialize with a dummy value
dribble_count = 0
is_ball_upward = False  # Flag to track upward motion

# Lists to store ball positions for plotting
ball_positions_x = []
ball_positions_y = []

# Create a new window for displaying player's pose
pose_window_name = 'Player Pose'
cv2.namedWindow(pose_window_name)
cv2.moveWindow(pose_window_name, 1600, 0)  # Adjust the window position

# Create a new window for displaying ball movement
ball_movement_window_name = 'Ball Movement'
cv2.namedWindow(ball_movement_window_name)
cv2.moveWindow(ball_movement_window_name, 800, 0)  # Adjust the window position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe Pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Threshold the image to get a binary mask of the yellow region
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations to enhance ball detection
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Predict the next state using the Kalman filter
            prediction = kalman.predict()

            # Correct the state based on the measured values
            measurement = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
            kalman.correct(measurement)

            # Draw the predicted and corrected positions on the frame
            cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(measurement[0]), int(measurement[1])), 5, (0, 255, 0), -1)

            # Draw the bounding box around the ball
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Count dribble if the ball moves upward (change this based on your video)
            if prev_position is not None and measurement[1] < prev_position[1] and not is_ball_upward:
                dribble_count += 1
                is_ball_upward = True
            elif prev_position is not None and measurement[1] > prev_position[1]:
                is_ball_upward = False

            prev_position = measurement

            # Store ball positions for plotting
            ball_positions_x.append(measurement[0][0])
            ball_positions_y.append(measurement[1][0])

    # Display dribble count on the video
    cv2.putText(frame, f'Dribbles: {dribble_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Tracking and Counting Dribbles', frame)

    # Display player's pose in a separate window
    pose_frame = np.zeros_like(frame)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = pose_frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(pose_frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow(pose_window_name, pose_frame)

    # Create a new window for displaying ball movement
    ball_movement_frame = np.zeros_like(frame)
    if ball_positions_x and ball_positions_y:
        ball_x, ball_y = int(ball_positions_x[-1]), int(ball_positions_y[-1])
        cv2.circle(ball_movement_frame, (ball_x, ball_y), 10, (0, 255, 255), -1)

    cv2.imshow(ball_movement_window_name, ball_movement_frame)

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
