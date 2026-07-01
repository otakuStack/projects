import os
import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_processor import FaceProcessor
from src.modules.blink import BlinkDetector
from src.modules.lighting import LightingAnalyzer
from src.modules.jitter import JitterAnalyzer

st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🛡️ Deepfake Video Call Detector Dashboard")
st.caption("Live, privacy-preserving local execution using interpretable visual artifacts.")

# Setup Layout Columns
col_vid, col_dash = st.columns([2, 1])

with col_dash:
    st.header("📊 Real-Time Diagnostic Analysis")
    risk_metric = st.empty()
    blink_metric = st.empty()
    light_metric = st.empty()
    jitter_metric = st.empty()

with col_vid:
    run_stream = st.checkbox("Initialize Webcam Interface", value=False)
    video_placeholder = st.empty()

# Instantiate core engines
processor = FaceProcessor()
blinker = BlinkDetector()
lighter = LightingAnalyzer()
jitterer = JitterAnalyzer()

if run_stream:
    cap = cv2.VideoCapture(0)
    
    while run_stream:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam stream disconnected.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        landmarks = processor.extract_landmarks(frame)
        
        # Default safety states
        risk_score = 10
        status_string = "✅ Clear / Secure"
        
        if landmarks:
            # 1. Process Signal Metrics
            ear, total_blinks = blinker.process(landmarks, w, h)
            light_variance = lighter.analyze_consistency(frame, landmarks, w, h)
            jitter_score = jitterer.calculate_jitter(landmarks)
            
            # 2. Simple Heuristic Signal Fusion (PRD Scoring Engine)
            # Example: excessive jitter or static abnormal lighting spikes risks
            if jitter_score > 3.5 or light_variance > 55:
                risk_score = 75
                status_string = "🚨 High Deepfake Risk Detected!"
            elif jitter_score > 2.0:
                risk_score = 45
                status_string = "⚠️ Anomalous Artifact Warning"

            # 3. Dynamic UI Updates
            risk_metric.metric("Overall Threat Index", f"{risk_score}%", status_string)
            blink_metric.info(f"👁️ Total Eye Blinks Measured: {total_blinks} (Current EAR: {ear:.2f})")
            light_metric.info(f"💡 Lighting Structure Variance: {light_variance:.2f}")
            jitter_metric.info(f"📉 Mechanical Landmark Jitter: {jitter_score:.2f}")
        else:
            risk_metric.metric("Overall Threat Index", "0%", "No Subject Face Detected")

        # Display Frame output
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame)
        time.sleep(0.01) # Yield to maintain thread responsiveness

    cap.release()