import warnings
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
import csv
import pygame
import winsound
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Suppress Protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize Pygame mixer
pygame.mixer.init()

# Function to log events
def log_event(event):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/events.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event])

# Function to plot events
def plot_events():
    if not os.path.exists('logs/events.csv'):
        print("No event log found.")
        return

    events = []
    timestamps = []

    with open('logs/events.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            timestamps.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
            events.append(row[1])

    # Creating color mapping for different states
    color_map = {'Active': 'green', 'Drowsy': 'orange', 'Sleeping': 'red'}
    colors = [color_map[event] for event in events]

    plt.figure(figsize=(14, 7))
    plt.scatter(timestamps, events, c=colors, s=100, label='Drowsiness States', alpha=0.6, edgecolors='w', linewidths=2)
    plt.plot(timestamps, events, linestyle='-', color='blue', alpha=0.4)

    plt.title("Drowsiness Detection States Over Time", fontsize=16)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("State", fontsize=12)
    
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))            # Tick every 3 hours
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adding legend and labels
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('logs/enhanced_events_plot.png')
    print("Plot saved as logs/enhanced_events_plot.png")

# Function to play sound alert
def sound_alert(play=True):
    if play:
        ringtone_file = 'ringg.mp3'
        if os.path.exists(ringtone_file):
            # Play the ringtone file
            pygame.mixer.music.load(ringtone_file)
            pygame.mixer.music.play()
            return
        else:
            # Default to normal beep
            winsound.Beep(1000, 1000)  # Frequency: 1000Hz, Duration: 1000ms
            # For Unix-based systems, use: os.system('play -nq -t alsa synth 1 sine 440')
    else:
        pygame.mixer.music.stop()

# Initialize the camera
cam = cv2.VideoCapture(0)

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize state variables
class State:
    ACTIVE = "Active"
    DROWSY = "Drowsy"
    SLEEPING = "Sleeping"

current_state = State.ACTIVE
previous_state = State.ACTIVE

# Use deques to store recent blink ratios
left_blink_ratios = deque(maxlen=30)  # Store last 30 frames (1 second at 30 fps)
right_blink_ratios = deque(maxlen=30)

# Function to compute Euclidean distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

# Function to determine eye aspect ratio
def eye_aspect_ratio(eye_landmarks):
    vertical_dist1 = compute(eye_landmarks[1], eye_landmarks[5])
    vertical_dist2 = compute(eye_landmarks[2], eye_landmarks[4])
    horizontal_dist = compute(eye_landmarks[0], eye_landmarks[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# Function to detect head tilt
def detect_head_tilt(face_landmarks):
    left_eye = np.array(face_landmarks[33])
    right_eye = np.array(face_landmarks[263])
    nose = np.array(face_landmarks[1])

    eye_line = np.append(right_eye - left_eye, 0)  # Make it 3D by appending 0
    nose_line = np.append(nose - left_eye, 0)      # Make it 3D by appending 0

    angle = np.degrees(np.arctan2(np.cross(eye_line, nose_line)[-1], np.dot(eye_line, nose_line)))
    return abs(angle) > 20  # Return 'True' if head is tilted more than 20 degrees

# Function to detect yawning
def detect_yawn(face_landmarks, hand_landmarks):
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    mouth_open = compute(upper_lip, lower_lip)
    thresh = 30  # Adjust threshold as needed
    
    if mouth_open > thresh or detect_mouth_covered(face_landmarks, hand_landmarks):
        return True
    else:
        return False

# Function to detect if mouth is covered by hands
def detect_mouth_covered(face_landmarks, hand_landmarks):
    mouth_center = np.mean([face_landmarks[13], face_landmarks[14]], axis=0)
    for hand_landmark in hand_landmarks:
        for lm in hand_landmark.landmark:
            hand_point = (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
            if compute(mouth_center, hand_point) < 50:  # Adjust threshold as needed
                return True
    return False

def get_state(left_ear, right_ear, yawning):
    avg_ear = (left_ear + right_ear) / 2

    if avg_ear < 0.18:
        return State.SLEEPING
    elif avg_ear < 0.22 or yawning:
        return State.DROWSY
    else:
        return State.ACTIVE

last_transition_time = time.time()
transition_cooldown = 5  # in seconds

# Function to save snapshot
def save_snapshot(frame):
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"snapshots/snapshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved as {filename}")

frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks]

        left_eye_landmarks = [landmarks[p] for p in [33, 160, 158, 133, 153, 144]]
        right_eye_landmarks = [landmarks[p] for p in [362, 385, 387, 263, 373, 380]]

        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)

        left_blink_ratios.append(left_ear)
        right_blink_ratios.append(right_ear)

        head_tilted = detect_head_tilt(landmarks)
        
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks

        yawning = detect_yawn(landmarks, hand_landmarks)
        current_state = get_state(left_ear, right_ear, yawning)

        # Check for state transition
        if current_state != previous_state and time.time() - last_transition_time > transition_cooldown:
            print(f"State transition: {previous_state} -> {current_state}")
            print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Head Tilted: {head_tilted}, Yawning: {yawning}")
            last_transition_time = time.time()
            previous_state = current_state

            # Log the event
            log_event(current_state)

            # Play sound alert if state is Sleeping
            if current_state is State.SLEEPING:
                sound_alert()
            else:
                sound_alert(play=False)

        # Display status on frame
        cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Head Tilted: {head_tilted}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawning: {yawning}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw facial landmarks
        for x, y in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.putText(frame, "Press 's' to save snapshot", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:          # Press 'Esc' to exit
        break
    elif key == ord('s'):  # Press 's' to save snapshot
        save_snapshot(frame)

cam.release()
cv2.destroyAllWindows()

# Plot events after the loop ends
plot_events()
