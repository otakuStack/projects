import numpy as np

class JitterAnalyzer:
    def __init__(self):
        self.prev_landmarks = None
        # Use stable core nose and chin landmarks to evaluate frame-to-frame micro-shifting
        self.ANCHOR_POINTS = [1, 4, 152, 10] 

    def calculate_jitter(self, landmarks):
        """Calculates structural motion delta between consecutive frames."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return 0.0

        deltas = []
        for idx in self.ANCHOR_POINTS:
            curr_pt = np.array([landmarks[idx].x, landmarks[idx].y])
            prev_pt = np.array([self.prev_landmarks[idx].x, self.prev_landmarks[idx].y])
            deltas.append(np.linalg.norm(curr_pt - prev_pt))

        self.prev_landmarks = landmarks
        # Return scaled mean frame shift velocity
        return float(np.mean(deltas) * 1000)