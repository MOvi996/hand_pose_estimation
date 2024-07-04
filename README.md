
# Hand pose estimation

Tracks a video stream of human hand and detect keypoints using Mediapipe. Then calculate angles and rotation matrices.

To run the code, first install the required packages:

    pip install requirements.txt 

Then run landmarks.py to detect landmarks on data/hand_rec.mov file (or set your own path in the code file).

    python landmarks.py

Calculate joint angles and rotation matrices using angles_and_mats.py.

    python angles_and_mats.py




