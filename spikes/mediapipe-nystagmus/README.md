# MediaPipe Nystagmus Detection Spike

**Feasibility spike** for deterministic on-device nystagmus measurement using MediaPipe FaceLandmarker.

## Goal

Answer: can nystagmus be measured reliably at 60fps with MediaPipe in the browser?

## How to run

### Option 1: Static server (recommended)
```bash
cd spikes/mediapipe-nystagmus
npx serve
```

Then open `http://localhost:3000` in Chrome/Edge.

### Option 2: Direct open
```bash
open index.html
```

**Note**: If you get CORS errors when loading the model, use `npx serve` or any HTTP server.

## Usage

1. **Webcam**: Click "Start Webcam" for live tracking
2. **Video**: Click "Load Video" to process a recorded clip (e.g. Dix-Hallpike at 60fps)
3. **Observe**:
   - Green overlay: tracked iris points
   - Red dots: computed iris centers
   - Bottom graph: horizontal (red) and vertical (blue) velocity vs time
   - Metrics: FPS, velocity, nystagmus frequency, latency, beats, direction, fatigability

## Algorithm architecture

### Iris tracking
- **Landmarks**: 468-472 (left iris), 473-477 (right iris)
- **Center**: Average of the 5 iris points per eye
- **Stabilization**: Average of both eyes to reduce noise

### Velocity computation
```
vx = (dx_pixels * PX_TO_DEG) / dt
vy = (dy_pixels * PX_TO_DEG) / dt
```
Where `PX_TO_DEG ≈ 0.05` (approximation: ~20px per visual degree)

**Limitation**: This conversion is a rough estimate. Real deg/s would require:
- Calibration against a known reference (e.g. target at fixed distance)
- Accounting for camera-eye distance and lens FOV

### Nystagmus detection
- **Fast phase**: Velocity threshold > 30°/s
- **Minimum interval between beats**: 300ms (avoids false positives)
- **Frequency**: `n_beats / total_duration`
- **Direction**: Average velocity over the last 30 frames
- **Latency**: Time from start to first beat
- **Fatigability**: Amplitude comparison between the first and second half of the video

## Dependencies

- **MediaPipe Tasks Vision** v0.10.18 (via CDN)
- **Model**: FaceLandmarker (float16, ~30MB)
- **Browser**: Chrome/Edge with WebGPU/WebGL2

**All on-device**: The video is NEVER uploaded to any server. Only the MediaPipe library is downloaded.

---

## FEASIBILITY VERDICT

**COMPLETE AFTER VALIDATING WITH A REAL DIX-HALLPIKE CLIP**

### Observed metrics

| Metric | Value | Target | Pass? |
|---------|-------|----------|----------|
| Effective FPS | _ TBD _ | ≥55 fps | ⏳ |
| Tracking stability | _ TBD _ | <5% dropped frames | ⏳ |
| Beat detection | _ TBD _ | Sensitivity >80% | ⏳ |
| Direction accuracy | _ TBD _ | Agreement with expert | ⏳ |
| Detection latency | _ TBD _ | <500ms from start | ⏳ |

### Feasibility analysis

#### Tier 1: Horizontal/vertical velocity on-device

**Is it feasible?** _PENDING VALIDATION_

**Factors in favor**:
- MediaPipe FaceLandmarker supports 478 landmarks including iris
- GPU acceleration in the browser (WebGPU/WebGL2)
- No network latency (all on-device)

**Known limitations**:
1. **FPS**: MediaPipe reports ~30-45fps on typical hardware with iris tracking. 60fps is ambitious.
2. **px→degrees conversion**: The current approximation (0.05°/px) is rough. Clinical use requires per-patient calibration.
3. **Iris robustness**:
   - Degrades with poor or uneven lighting
   - Closed/partially closed eyelids cause dropout
   - Contact lenses can confuse the tracker
4. **Temporal aliasing**: At 30fps, nystagmus >1.5Hz may alias. 60fps is desirable.
5. **Phase detection**: The current algorithm (velocity threshold) is basic. Fast slow-phases require acceleration/jerk analysis.

#### Tier 2: Torsion (ocular rotation)

**Is it feasible?** **NOT in this iteration**

**Reasons**:
1. MediaPipe FaceLandmarker **does not provide torsion directly**
2. It would require:
   - Tracking iris features (crypt patterns)
   - Or tracking scleral blood vessels
   - Both need additional image processing (OpenCV.js, custom shaders)
3. Significant algorithmic complexity (image registration, optical flow)
4. Performance: probably <15fps with torsion

**Alternative for torsion**: Use dedicated video-oculography (hardware) or research algorithms (e.g. iris pattern matching with deep learning)

### Recommendation to the team

**For Tier 1 (V/H on-device)**:
- ✅ **Feasible with caveats** if the following is accepted:
  - Realistic FPS: 30-45fps (not 60fps)
  - Manual calibration for px→degrees conversion
  - Controlled lighting conditions
  - Validation against an expert on ≥20 clips

- 🔧 **Improvements needed for production**:
  1. Per-patient calibration (target at a known distance)
  2. Kalman filter to smooth the iris trajectory
  3. Slow/fast phase detection with jerk analysis (derivative of acceleration)
  4. Dropout handling (eyelids, blinks) with interpolation
  5. Formal clinical validation (sensitivity/specificity vs VOG gold standard)

**For Tier 2 (torsion)**:
- ❌ **Not feasible with MediaPipe alone**
- 🔬 Requires a separate spike with:
  - OpenCV.js for optical flow
  - Iris registration algorithm (e.g. phase-based optical flow)
  - Validation against artificial torsion (synthetic video with known rotation)

### Suggested next steps

1. **Validate this spike** with a real Dix-Hallpike clip at 60fps
2. **Measure real FPS** on target hardware (clinician's laptop, tablet)
3. **Compare** computed metrics vs expert interpretation
4. If FPS < 55: evaluate quality/speed trade-off (fewer landmarks, lite model)
5. If tracking is unstable: consider a Kalman filter or moving median

### Conclusion

**Tier 1 (V/H)**: **CONDITIONAL FEASIBILITY** — works under controlled conditions, but requires calibration and clinical validation before diagnostic use.

**Tier 2 (torsion)**: **NOT FEASIBLE** with MediaPipe alone. Requires an additional computer-vision stack and probably won't reach real time.

**Recommendation**: Proceed with Tier 1 as MVP, validate with ≥20 real clips, and evaluate whether the accuracy justifies deployment. Tier 2 remains future work or is delegated to dedicated hardware.

---

## Technical notes

### MediaPipe API used
```javascript
FaceLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath: "...",
    delegate: "GPU"
  },
  runningMode: "VIDEO",
  numFaces: 1,
  outputFaceBlendshapes: false,
  outputFacialTransformationMatrixes: false
});

faceLandmarker.detectForVideo(videoElement, timestampMs);
```

### Iris landmarks (478 total)
- **468**: Left iris center
- **469-472**: Left iris contour
- **473**: Right iris center
- **474-477**: Right iris contour

### Privacy
✅ All on-device. Zero video/PII upload to servers.

### Spike limitations
- Code not optimized for production (it is a spike)
- Hardcoded detection threshold (30°/s)
- Approximate px→degrees conversion
- No robust error handling
- No dropout interpolation

---

**Author**: Spike generated for Clinibrium VertigoDx
**Date**: July 2026
**Status**: ⏳ Pending validation with a real clip
