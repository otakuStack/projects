import numpy as np

class BlinkDetector:
    def __init__(self):
        # MediaPipe landmark indices for left and right eyes
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.EAR_THRESHOLD = 0.22 # Below this, eye is considered closed
        self.blink_count = 0
        self.eye_closed = False

    def _calculate_ear(self, landmarks, eye_indices, width, height):
        # Grab landmarks and scale to pixel coordinates
        pts = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in eye_indices]
        
        # EAR Formula: ||p2-p6|| + ||p3-p5|| / (2 * ||p1-p4||)
        vertical_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_2 = np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])
        
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def process(self, landmarks, width, height):
        """Tracks rolling blink statuses."""
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, width, height)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, width, height)
        avg_ear = (left_ear + right_ear) / 2.0

        # State machine to detect a distinct blink down-up motion
        if avg_ear < self.EAR_THRESHOLD:
            self.eye_closed = True
        else:
            if self.eye_closed:
                self.blink_count += 1
                self.eye_closed = False
                
        return avg_ear, self.blink_count