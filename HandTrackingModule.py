import cv2
import mediapipe as mp

class HandDetector:
    def __init__(
        self, 
        staticImage=False,
        maxHandsNum=2,
        minDetectionConfidence=0.5,
        minTrackingConfidence=0.5):

        self.staticImage = staticImage
        self.maxHandsNum = maxHandsNum
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(staticImage, maxHandsNum, minDetectionConfidence, minTrackingConfidence)
        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyle = mp.solutions.drawing_styles
        """
        MediaPipe Hands processes an RGB image and returns the hand landmarks and
        handedness (left v.s. right hand) of each detected hand.

        Args:
        staticImage: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream.
        maxHandsNum: Maximum number of hands to detect.
        minDetectionConfidence: Minimum confidence value ([0.0, 1.0]) for hand
        detection to be considered successful.
        minTrackingConfidence: Minimum confidence value ([0.0, 1.0]) for the
        hand landmarks to be considered tracked successfully.
        """

    def detectHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if draw:
            if (self.results.multi_hand_landmarks):
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mpDrawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyle.get_default_hand_landmarks_style(),
                        self.mpDrawingStyle.get_default_hand_connections_style())
        return img
    """
    Processes an image and returns that image with drawn hand landmarks or not (depends on draw parameter)
    Args:
        img: source image
        draw: draw hand landmarks or don't
    """
    def getLandmarksPosition(self, img, handNo):
        handLandmarksList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            imgHeight, imgWidth, imgChannel = img.shape
            for lm in hand.landmark:
                realLMX = int(lm.x * imgWidth)
                realLMY = int(lm.y * imgHeight)
                handLandmarksList.append([realLMX, realLMY])
        return handLandmarksList
    """
    Return a matrix of position of each hand landmarks of which choosen hand
    Args:
        img: source image
        handNo: hand number
    """
        
