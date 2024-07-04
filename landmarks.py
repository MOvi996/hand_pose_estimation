############
# code partially taken from https://medium.com/featurepreneur/mediapipe-pose-dectection-in-images-and-videos-31a583d5a7fb
############ 

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the video file
cap = cv2.VideoCapture('data/hand_rec.mov')

# Get the video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('data/hand_annotated.mp4', fourcc, fps, (width, height))

all_landmarks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame)
    frame_landmarks = []

    # Draw the hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                      mp.solutions.drawing_styles.get_default_hand_connections_style())
            
            frame_landmarks.append([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
    
    all_landmarks.append(frame_landmarks)

    out_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(out_frame)

all_landmarks = np.array(all_landmarks, dtype=object)
np.save('data/hand_landmarks.npy', all_landmarks)


cap.release()
out.release()
hands.close()
cv2.destroyAllWindows()

