import cv2


class _SimpleLandmark:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class FaceProcessor:
    def __init__(self):
        self.face_mesh = None
        self.face_cascade = None
        self.using_fallback = True

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            self.face_cascade = None

    def _build_fallback_landmarks(self, frame, x, y, w, h):
        height, width = frame.shape[:2]
        # Create list of 468 default points
        landmarks = [_SimpleLandmark(0.5, 0.5) for _ in range(468)]

        def set_point(index, px, py):
            landmarks[index] = _SimpleLandmark(px / width, py / height)

        left_eye_center = (x + w * 0.30, y + h * 0.42)
        right_eye_center = (x + w * 0.70, y + h * 0.42)
        nose_center = (x + w * 0.50, y + h * 0.58)
        chin_center = (x + w * 0.50, y + h * 0.86)

        for idx, point in [
            (362, (left_eye_center[0] - w * 0.02, left_eye_center[1] - h * 0.01)),
            (385, (left_eye_center[0] + w * 0.00, left_eye_center[1] - h * 0.02)),
            (387, (left_eye_center[0] + w * 0.02, left_eye_center[1] - h * 0.01)),
            (263, (left_eye_center[0] - w * 0.02, left_eye_center[1] + h * 0.01)),
            (373, (left_eye_center[0] + w * 0.00, left_eye_center[1] + h * 0.02)),
            (380, (left_eye_center[0] + w * 0.02, left_eye_center[1] + h * 0.01)),
            (33, (right_eye_center[0] - w * 0.02, right_eye_center[1] - h * 0.01)),
            (160, (right_eye_center[0] + w * 0.00, right_eye_center[1] - h * 0.02)),
            (158, (right_eye_center[0] + w * 0.02, right_eye_center[1] - h * 0.01)),
            (133, (right_eye_center[0] - w * 0.02, right_eye_center[1] + h * 0.01)),
            (153, (right_eye_center[0] + w * 0.00, right_eye_center[1] + h * 0.02)),
            (144, (right_eye_center[0] + w * 0.02, right_eye_center[1] + h * 0.01)),
            (1, (nose_center[0], nose_center[1] - h * 0.02)),
            (4, (nose_center[0], nose_center[1] + h * 0.01)),
            (10, (nose_center[0] - w * 0.01, nose_center[1] + h * 0.02)),
            (152, (chin_center[0], chin_center[1] + h * 0.01)),
        ]:
            set_point(idx, point[0], point[1])

        return landmarks

    def extract_landmarks(self, frame):
        """Returns normalized landmark data if a face is detected, else None."""
        # Route to MediaPipe if it successfully launched
        if self.face_mesh is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None

        # Route to Haar Cascade fallback otherwise
        if self.face_cascade is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return None

        # Pick the largest face area found
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        return self._build_fallback_landmarks(frame, x, y, w, h)