

Real-Time Deepfake Video Call Detector

Version: 1.0
Owner: [Your Name]
Status: Draft
Target: Portfolio / Student Research Project


1. Problem Statement

Deepfake-driven "CEO fraud" and impersonation scams are rising — attackers use real-time face-swap or voice-clone tech during video calls to impersonate executives and authorize fraudulent wire transfers, share credentials, or manipulate employees. Existing detection tools are mostly post-hoc (analyzing recorded video) rather than real-time, and are not accessible to individuals or small orgs without enterprise security budgets.

Goal: Build a lightweight, real-time tool that flags likely deepfake video during a live call using observable visual artifacts — without needing cloud-scale compute or proprietary datasets.


2. Objectives


Detect deepfake video in near real-time (target: <2 second latency per analysis window)
Use interpretable, artifact-based signals (not a black-box only) so users understand why something was flagged
Run locally or as a lightweight browser/desktop overlay — no need to route video through third-party servers (privacy-preserving)
Produce a portfolio-ready demo: dashboard showing live confidence score + flagged artifacts



3. Non-Goals (v1)


Not building a production-grade enterprise security product
Not aiming for 99%+ accuracy against state-of-the-art deepfakes (that requires much larger datasets/compute)
Not doing voice deepfake detection in v1 (video-only scope; voice can be v2)
Not integrating directly into Zoom/Teams APIs in v1 (use screen-capture or webcam-passthrough as MVP)



4. Target Users


Security-conscious individuals/employees worried about impersonation scams
Small businesses without enterprise deepfake-detection tools
Recruiters/evaluators viewing this as a portfolio/research project
(Secondary) Researchers wanting an open, interpretable baseline detector



5. Core Detection Signals (v1 scope)

SignalWhat It MeasuresWhy It WorksBlink rate & patternFrequency and naturalness of eye blinksEarly deepfake models under/over-generate blinks; real humans blink ~15-20 times/min with natural irregularityLighting consistencyWhether face lighting/shadows match the rest of the sceneFace-swap models often render a face lit differently than the background/neckHead pose vs. facial landmark jitterMicro-jitter or unnatural smoothness in landmark tracking across framesSynthetic faces often show frame-to-frame landmark instability under fast head motionBoundary/blending artifactsEdge inconsistency around jawline, hairline, earsFace-swap seams often show subtle color/texture mismatchTemporal consistency (optical flow)Whether motion between frames is physically plausibleGenerative models can produce flicker or unnatural frame-to-frame texture shifts


v1 will implement blink detection + lighting consistency + landmark jitter as the core 3 signals (feasible within a few weeks). Boundary artifacts and optical flow can be v1.5/v2 stretch goals.




6. User Flow


User starts a video call (or opens the tool alongside a call) and enables the detector
Tool accesses webcam feed (or screen-share region) via browser/desktop app
Every N frames (e.g., every 1 second), the tool:

Runs face detection + landmark extraction
Computes blink rate over a rolling window
Computes lighting consistency score between face region and background
Computes landmark jitter score



Combines signals into a single "Deepfake Risk Score" (0–100)
Displays a live indicator (green/yellow/red) with a breakdown of which signal(s) triggered
Optionally logs a session report at call end (for post-call review)



7. Success Metrics


Detection accuracy on a labeled test set (real vs. known deepfake clips) — target 75-85% for v1 (be honest about limitations in your writeup; this is expected for an interpretable, non-deep-learning baseline)
Latency — under 2 seconds per analysis cycle on a standard laptop (no GPU requirement for v1)
False positive rate on real video under varied lighting — should be documented and minimized
Demo quality — clear, explainable UI showing why a flag was raised (important for portfolio/interview discussions)



8. Technical Architecture

[Webcam / Screen Capture]
        ↓
[Frame Sampler] — grabs 1 frame every ~200-300ms
        ↓
[Face Detection + Landmarks] — MediaPipe Face Mesh / dlib
        ↓
   ┌─────────────┬─────────────────┬────────────────┐
   ↓             ↓                 ↓
[Blink Module] [Lighting Module] [Jitter Module]
   ↓             ↓                 ↓
        [Signal Fusion / Scoring Engine]
                    ↓
        [Risk Score + UI Overlay/Dashboard]
                    ↓
        [Optional: Session Log / Report]


9. Tech Stack (Recommended)

LayerToolFace detection & landmarksMediaPipe Face Mesh (Python or JS) or dlib (Python)Blink detectionEye Aspect Ratio (EAR) algorithm on eye landmarksLighting analysisOpenCV — histogram/luminance comparison between face ROI and surrounding regionJitter analysisFrame-to-frame landmark position variance (custom, using OpenCV/NumPy)Backend/processingPython (OpenCV, MediaPipe, NumPy)Frontend/UIStreamlit (fastest for demo) OR React + WebRTC if you want a polished browser toolPackagingLocal desktop app (Python + OpenCV window) for MVP; browser extension as stretch goal

Why this stack: All open-source, all runnable without GPU, and every component is explainable in an interview ("I used Eye Aspect Ratio for blink detection because...").


10. Milestones (Suggested Timeline)

WeekDeliverable1Face detection + landmark extraction working on live webcam feed2Blink detection module (EAR-based) + rolling blink-rate calculation3Lighting consistency module + landmark jitter module4Signal fusion scoring engine + basic UI (Streamlit or OpenCV overlay)5Testing against sample real vs. deepfake video datasets (e.g., FaceForensics++ subset, publicly available for research)6Polish UI, write documentation, record demo video, publish to GitHub + LinkedIn


11. Known Limitations (be upfront about these — it builds credibility)


Cannot reliably detect the newest, high-quality real-time deepfakes (this is an active research arms race)
Lighting/blink signals can produce false positives in poor webcam conditions or unusual lighting
v1 does not analyze audio, which is a major gap for full CEO-fraud-style detection
Requires face to be clearly visible; doesn't work well with masks, heavy makeup, or extreme angles



12. Stretch Goals (v2+)


Voice deepfake detection (spectral + prosody analysis) fused with video score
Browser extension for direct Zoom/Google Meet overlay
Lightweight CNN classifier trained on public deepfake datasets as a 4th signal
Multi-face support for group calls
