# Importing the Essential Libraries
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initializing MediaPipe Pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Initializing The Default Web Camera
cap = cv2.VideoCapture(0)

# Creating Two Drawing Specifications
drawspec1 = mp_drawing.DrawingSpec(thickness=4,circle_radius=5,color=(0,0,255))
drawspec2 = mp_drawing.DrawingSpec(thickness=4,circle_radius=8,color=(0,255,0))

# Checking the camera is opened or not
if not cap.isOpened():
    print("Error: Camera not found or cannot be opened.")
    exit()

# Initializing Some Variables
jump_count = 0
leg_in_air = False
right_var = 540
left_var = 540

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(600,600))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,drawspec1, drawspec2)

    if results.pose_landmarks:
        # Ankle landmark indices may vary depending on the model version, adjust them as needed.
        left_ankle_landmark = results.pose_landmarks.landmark[27]  # Example for left ankle
        right_ankle_landmark = results.pose_landmarks.landmark[28]  # Example for right ankle

        # Extract 3D coordinates for left ankle
        left_ankle_x = int(left_ankle_landmark.x * frame.shape[1])
        left_ankle_y = int(left_ankle_landmark.y * frame.shape[0])
        left_ankle_z = left_ankle_landmark.z

        # Extract 3D coordinates for right ankle
        right_ankle_x = int(right_ankle_landmark.x * frame.shape[1])
        right_ankle_y = int(right_ankle_landmark.y * frame.shape[0])
        right_ankle_z = right_ankle_landmark.z

        cv2.putText(frame, f"Right Y:{right_ankle_y}",(50,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame, f"Left Y:{left_ankle_y}",(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)

        # Draw circles at the detected ankle positions
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 8, (0, 0, 255), -1)

        if left_ankle_y < left_var and right_ankle_y < right_var :
             if not leg_in_air:
                leg_in_air = True
                jump_count = jump_count +1
        else:
            leg_in_air = False

    h,w,c = image.shape
    imgBlank = np.zeros([h,w,c])
    imgBlank.fill(255)
    mp_drawing.draw_landmarks(imgBlank, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, drawspec1, drawspec2)

    cv2.putText(imgBlank, f"Jumps:{jump_count}",(50,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)

    cv2.imshow("Main Frame Detection", frame)
    cv2.imshow("Extracrted Detection", imgBlank)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
