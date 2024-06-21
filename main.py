import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np


angle_min = []
angle_min_hip = []
angle_min_ankle = []
cap = cv2.VideoCapture(0)

# Set the desired width and height for resizing
desired_width = 1280
desired_height = 720

# Initialize a flag to indicate whether a heel raise has been counted in the current cycle
counted = False
squat_counted = False

# Curl counter variables
squats_counter = 0
walking_counter = 0
Abduction_counter = 0
HR_counter = 0
RC_counter = 0
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
min_ang_ankle = 0
max_ang_ankle = 0
stage = None
delay_threshold = 30  # Number of frames to wait before counting another heel raise
delay_counter = 0

# Adjusted angle thresholds for heel raise detection
lower_threshold = 165  # Lowered threshold for starting the heel raise
upper_threshold = 170  # Lowered threshold for completing the heel raise

knee_angle_range = (170, 180)  # Example range for knee angle
hip_angle_range = (170, 180)   # Example range for hip angle

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video_2.mp4', fourcc, 24, (desired_width, desired_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        if not ret:
            print("Failed to grab a frame")
            break

        # Resize the frame to the desired resolution
        image = cv2.resize(frame, (desired_width, desired_height))

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect poses in the frame
        results = pose.process(image_rgb)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            """elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            """

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

            # Calculate angle
            # angle = calculate_angle(shoulder, elbow, wrist)

            angle_knee = calculate_angle(hip, knee, ankle)  # Knee joint angle
            angle_knee = round(angle_knee, 2)

            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_hip = round(angle_hip, 2)

            angle_ankle = calculate_angle(knee, ankle, foot)
            angle_ankle = round(angle_ankle, 2)

            hip_angle = 180 - angle_hip
            knee_angle = 180 - angle_knee
            ankle_angle = 180 - angle_ankle

            angle_min.append(angle_knee)
            angle_min_hip.append(angle_hip)
            angle_min_ankle.append(angle_ankle)
            # print(angle_knee)
            # Visualize angle
            """cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )"""

            cv2.putText(image, str(angle_knee),
                        tuple(np.multiply(knee, [1500, 800]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(angle_hip),
                        tuple(np.multiply(hip, [1500, 800]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(angle_ankle),
                        tuple(np.multiply(ankle, [1500, 800]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Reverse Crunch counter logic
            # if angle_knee > 55:
            #     stage = "up"
            # elif angle_knee <= 55 and stage == 'up':
            #     stage = "down"
            #     RC_counter += 1
            #     print(f"Reverse Crunch Count: {RC_counter}")
            #     min_ang = min(angle_min)
            #     max_ang = max(angle_min)
            #
            #     min_ang_ankle = min(angle_min_ankle)
            #     min_ang_ankle = max(angle_min_ankle)
            #
            #     min_ang_hip = min(angle_min_hip)
            #     max_ang_hip = max(angle_min_hip)
            #
            #     print(min(angle_min), " _ ", max(angle_min))
            #     print(min(angle_min_hip), " _ ", max(angle_min_hip))
            #     angle_min = []
            #     angle_min_hip = []
            #     angle_min_ankle = []

            # Squat counter logic
            if angle_knee > 150:
                squat_stage = "up"
                squat_counted = False
            elif angle_knee <= 150 and squat_stage == 'up':
                squat_stage = "down"
                squats_counter += 1
                squat_counted = True
                print(f"Squats Count: {squats_counter}")
                min_ang = min(angle_min)
                max_ang = max(angle_min)

                min_ang_ankle = min(angle_min_ankle)
                min_ang_ankle = max(angle_min_ankle)

                min_ang_hip = min(angle_min_hip)
                max_ang_hip = max(angle_min_hip)

                angle_min = []
                angle_min_hip = []
                angle_min_ankle = []


            # Heel Raise counter logic
            # Heel Raise counter logic
            # if angle_ankle > 156 and not counted:
            #     heel_raise_stage = "down"
            # elif angle_ankle <= 156 and heel_raise_stage == 'down' and not counted:
            #     heel_raise_stage = "up"
            #     HR_counter += 1
            #     counted = True  # Set the flag to indicate a heel raise has been counted
            #     print(f"Heel Raises Count: {HR_counter}")
            #     min_ang = min(angle_min)
            #     max_ang = max(angle_min)
            #
            #     min_ang_ankle = min(angle_min_ankle)
            #     max_ang_ankle = max(angle_min_ankle)
            #
            #     min_ang_hip = min(angle_min_hip)
            #     max_ang_hip = max(angle_min_hip)
            #
            #     angle_min = []
            #     angle_min_hip = []
            #     angle_min_ankle = []
            #
            #     # Reset the counted flag for heel raise at the end of the cycle
            #     if heel_raise_stage == "up" and angle_ankle < 170:
            #         counted = False

            # Walking counter logic
            # if angle_ankle > 170 and angle_ankle < 169.3 and angle_knee >= 180:
            #     stage = "up"
            # elif angle_ankle <= 170 and angle_ankle >= 169.3 and stage == 'up':
            #     stage = "down"
            #     walking_counter += 1
            #     print(f"Walking Count: {walking_counter}")
            #     min_ang = min(angle_min)
            #     max_ang = max(angle_min)
            #
            #     min_ang_ankle = min(angle_min_ankle)
            #     min_ang_ankle = max(angle_min_ankle)
            #
            #     min_ang_hip = min(angle_min_hip)
            #     max_ang_hip = max(angle_min_hip)
            #
            #     print(min(angle_min), " _ ", max(angle_min))
            #     print(min(angle_min_hip), " _ ", max(angle_min_hip))
            #     angle_min = []
            #     angle_min_hip = []
            #     angle_min_ankle = []



        except:
            pass

        # Render squat counter
        # Setup status box
        cv2.rectangle(image, (20, 20), (435, 200), (0, 0, 0), -1)

        # Rep data
        """cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)"""
        cv2.putText(image, "Squats : " + str(squats_counter),
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Heel Raise : " + str(HR_counter),
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "Walking : " + str(walking_counter),
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "Reverse Crunch : " + str(RC_counter),
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        """cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)"""
        """cv2.putText(image, stage, 
                    (10,120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)"""

        # # Knee angle:
        # """cv2.putText(image, 'Angle', (65,12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)"""
        # cv2.putText(image, "Knee-joint angle : " + str(min_ang),
        #             (30, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # Hip angle:
        # cv2.putText(image, "Hip-joint angle : " + str(min_ang_hip),
        #             (30, 140),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #
        #
        # # Ankle angle
        # cv2.putText(image, "Ankle-joint angle : " + str(angle_min_ankle),
        #             (30, 180),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2)
                                  )

        out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # cap.release()
            # out.release()
            # cv2.destroyAllWindows()
            break

        # print(angle_ankle)


    cap.release()
    out.release()
    cv2.destroyAllWindows()
