"""
This application adjust your PC volume by hand gesture
"""

import cv2
import HandTrackingModule as ht
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = ht.cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

hands = ht.HandDetector()

# Adjust system volume with pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)

handss = ht.HandDetector()
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Cannot open camera.")
        continue
    hands.detectHands(img)

    landmarks = hands.getLandmarksPosition(img, 0)
    
    if landmarks:
        # Thumb position in image frame
        thb = landmarks[4]
        # Index finger position in image frame
        idf = landmarks[8]
        cv2.line(img, thb, idf, (255, 0, 0), 2)
        # Distance between thumb and index finger
        thbIdfDistance = math.dist(thb, idf)
        print("Distance between thumb and index finger: ", thbIdfDistance)
        # Hand: 0 - 150
        # Volume range in pycaw: -65.25 - 0.0
        volumeLv = np.interp(thbIdfDistance, [0, 150], list(volume.GetVolumeRange())[0: 2])
        volume.SetMasterVolumeLevel(volumeLv, None)

    cv2.imshow("Camera with hands tracking.", cv2.flip(img, 1))
    cv2.waitKey(1)