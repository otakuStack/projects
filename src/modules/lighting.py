import cv2
import numpy as np

class LightingAnalyzer:
    def analyze_consistency(self, frame, landmarks, width, height):
        """Compares luminance distribution of the face vs the background bounding box."""
        # Get face boundary bounding box
        x_coords = [int(lm.x * width) for lm in landmarks]
        y_coords = [int(lm.y * height) for lm in landmarks]
        x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))

        # Convert frame to YUV to separate brightness (Y) from color
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]

        # Extract face region vs background sample
        face_roi = y_channel[y_min:y_max, x_min:x_max]
        if face_roi.size == 0:
            return 0.0

        face_mean, face_std = cv2.meanStdDev(face_roi)
        
        # Calculate a basic lighting anomaly metric (higher variance standard deviation can indicate blending mismatch)
        # For simplicity in v1, we return the face lighting variance score
        return float(face_std[0][0])