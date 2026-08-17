package com.example.insolescanplan2.docscan

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.roundToInt

// Matches the source SDK's own IMAGE_ANALYSIS_SCALE_WIDTH - the detection
// algorithm's fixed-pixel kernel sizes were tuned against frames this wide.
// Not an algorithm change - matching the original's own assumption.
private const val ANALYSIS_TARGET_WIDTH = 400

// How many color readings the Sample button collects before auto-stopping -
// several frames rather than one, so a single noisy/motion-blurred frame
// doesn't become the whole reference.
private const val COLOR_SAMPLE_TARGET_COUNT = 20

// Minimum time between auto-captures during a scan session - a guard
// against the same held pose re-triggering a second capture if the
// orientation signature jitters across a bucket boundary right after a
// capture, not a meaningful UX pacing choice.
private const val CAPTURE_COOLDOWN_MS = 1500L

// How long a genuinely new angle has to be held before it's actually
// captured - the corner stabilizer already means the QUAD is holding still,
// but the user's hand is very likely still in frame right when they stop
// turning the foam. This delay, with a big on-screen countdown, is the
// user's cue to pull their hand out of the way before the shutter fires.
private const val HOLD_STILL_MS = 2000L

// Padding added around the detected quad's bounding box before cropping a
// captured photo, as a fraction of that box's own width/height - keeps
// detection jitter from clipping a real edge of the foam out of the saved
// crop.
private const val CROP_MARGIN_RATIO = 0.1

private const val CAPTURE_LOG_TAG = "InsoleCapture"

// Reset to a minimal baseline (see md/status-2026-08-14.md) after a long
// chain of additions (temporal stabilization, color confirmation, several
// geometric guards) didn't resolve unreliable detection, and it became hard
// to tell whether the base algorithm or one of the additions was at fault.
// This is now just: live preview, the original algorithm's raw per-frame
// result, drawn directly - nothing smoothed, filtered, or confirmed.
@Composable
fun DocScanScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val openCvReady = remember { OpenCVLoader.initLocal() }

    // Explicitly set rather than relying on whatever PreviewView's actual
    // default is - mapAnalysisPointToScreen's math assumes a specific
    // behavior (uniform scale to COVER the view, cropping the overflow,
    // centered) and real testing showed that assumption didn't hold even
    // after ruling out every other explanation (Preview/Analysis stream
    // mismatch, rotation, resolution negotiation) - pinning this explicitly
    // removes the guesswork about what's actually being rendered.
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }
    val nativeBridge = remember { OpenCvNativeBridge() }
    val cornerStabilizer = remember { CornerStabilizer() }
    // The analyzer below runs a genuinely expensive OpenCV pipeline (Canny,
    // morphology, contour-finding) every frame - kept off the main/UI thread
    // so it never competes with normal recomposition work there. Not part of
    // the original algorithm either way - purely where it executes.
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Screen-space corners of the CURRENT displayed result, already mapped
    // into the overlay Canvas's own pixel coordinates - null means nothing
    // to show this frame. Raw per-frame detection when the stabilizer is
    // off, the stabilizer's smoothed result when it's on.
    var detectedCorners by remember { mutableStateOf<List<Offset>?>(null) }
    // Debug readout: what resolution we asked CameraX for vs what the
    // analysis Mat actually comes in as, plus the live preview view's own
    // pixel size - mapAnalysisPointToScreen assumes a fixed 90-degree
    // rotation between these, and if the ACTUAL analysis frame doesn't have
    // the dimensions that assumption expects, the overlay would be
    // systematically distorted on one axis (reported symptom: top/bottom of
    // a real rectangle not covered by the green outline) even with every
    // detection guard off, since guards only accept/reject candidates, they
    // never touch where points get drawn.
    var debugFrameInfo by remember { mutableStateOf("") }
    // Smooths the jittery raw per-frame detection into a steady result -
    // defaults ON, unlike the shape guards below, since it doesn't change
    // WHAT gets accepted as a detection, only how steady it looks once
    // accepted. Suspected but NOT confirmed as the source of the
    // "always-long-object" symptom - if a spuriously long environmental
    // artifact (a floorboard seam, a shadow) is being detected raw and
    // consistently, the stabilizer would faithfully lock onto and display
    // it just as confidently as a real object, which could make a
    // pre-existing raw-detection issue look worse/stickier than it already
    // was, rather than being the root cause itself. Worth comparing
    // behavior with this off vs on against the same real test conditions.
    var stabilizerEnabled by remember { mutableStateOf(true) }

    // The geometric guards inside OpenCvNativeBridge, reapplied one at a
    // time - all default OFF (universal any-4-edge scan is the baseline),
    // toggleable independently so each one's effect can be tested in
    // isolation rather than several at once.
    var edgeMarginGuardEnabled by remember { mutableStateOf(false) }
    var oppositeSideGuardEnabled by remember { mutableStateOf(false) }
    var minAreaGuardEnabled by remember { mutableStateOf(false) }
    var aspectRatioGuardEnabled by remember { mutableStateOf(false) }
    // Not a rejection guard - a SELECTION preference among survivors ("pick
    // whichever candidate sits closest to frame center"). Not part of the
    // original algorithm at all.
    var centerPriorityEnabled by remember { mutableStateOf(false) }

    // The Camera instance returned by bindToLifecycle - only available once
    // binding actually completes (async, inside the listener below), so the
    // torch toggle button has to read whatever's here at click time rather
    // than assuming it's ready immediately.
    var boundCamera by remember { mutableStateOf<Camera?>(null) }
    var flashEnabled by remember { mutableStateOf(false) }

    // Live color reading from whatever's currently detected, every frame -
    // shown on screen for feedback and fed into collectedSamples below while
    // sampling is active.
    var sampledColor by remember { mutableStateOf<SampledColor?>(null) }
    // True between tapping Sample and either reaching
    // COLOR_SAMPLE_TARGET_COUNT or tapping it again to stop early.
    var colorSamplingActive by remember { mutableStateOf(false) }
    var collectedSamples by remember { mutableStateOf(listOf<SampledColor>()) }
    // The averaged reference color, set by Lock - null until the user has
    // actually locked one in, at which point matchesLockedColor (see
    // ColorCheck.kt) starts being usable as a real guard instead of nothing
    // to compare against.
    var lockedColor by remember { mutableStateOf<SampledColor?>(null) }
    // Locking auto-enables this, but it stays a separate toggle so
    // enforcement can be paused/resumed without re-sampling.
    var colorGuardEnabled by remember { mutableStateOf(false) }

    // Guided multi-angle capture session - see md/plan for the full design.
    // While active, the tuning/guard UI is replaced with progress + prompt
    // text and a Stop button, and the corner stabilizer is forced on
    // (auto-capture needs its "held roughly still" debounce regardless of
    // the tuning toggle's own state).
    var sessionActive by remember { mutableStateOf(false) }
    val captureCoverage = remember { CaptureCoverage() }
    // Which cell the most recent successful capture landed in - a cell
    // already holding one capture only gets a second (the presumed
    // 180-degree twin) if this does NOT match its own cell, i.e. the two
    // captures of the same cell can't be consecutive.
    var lastCapturedCell by remember { mutableStateOf<Triple<Int, Int, Int>?>(null) }
    var coverageCount by remember { mutableStateOf(0) }
    var sessionPrompt by remember { mutableStateOf("") }
    // Internal bookkeeping for the analyzer callback (single-threaded via
    // cameraExecutor, so plain state without extra synchronization is safe)
    // - not read anywhere the UI needs it directly.
    var isCapturing by remember { mutableStateOf(false) }
    var lastCaptureTimeMs by remember { mutableStateOf(0L) }
    var captureSession by remember { mutableStateOf<CaptureSession?>(null) }
    // Which not-yet-covered cell is currently being held, and since when -
    // reset any time tracking is lost or the pose jumps to a different new
    // cell, so the hold has to be continuous. holdStillCountdown is the
    // UI-facing seconds-remaining derived from these each frame (null =
    // nothing counting down right now).
    var pendingCell by remember { mutableStateOf<Triple<Int, Int, Int>?>(null) }
    var pendingSince by remember { mutableStateOf(0L) }
    var holdStillCountdown by remember { mutableStateOf<Int?>(null) }

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    // Scrollable, and the camera Box below gets a FIXED height rather than
    // weight(1f) - with ~10 toggle buttons plus status/debug text, the
    // button area's natural height could exceed the screen, and weight(1f)
    // would have shrunk the camera preview to make room rather than letting
    // the extra content scroll - exactly the "buttons block the camera
    // view" complaint.
    Column(modifier = modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        if (!openCvReady) {
            Text("OpenCV failed to load")
            return@Column
        }
        if (!hasPermission) {
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Grant camera permission")
            }
            return@Column
        }

        DisposableEffect(lifecycleOwner) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            // Wait for previewView to actually be attached/laid out (like
            // the source SDK's viewFinder.post { ... }) before binding -
            // no longer needed for computing the target resolution (that's
            // fixed now), but still worth keeping so surfaceProvider isn't
            // handed out before the view is ready.
            previewView.post {
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    // FIXED target, deliberately NOT derived from
                    // previewView's live width/height (that dependency was
                    // fragile - as more buttons got added over time, the
                    // camera box's exact pixel size at bind time kept
                    // shifting, which pushed CameraX toward different
                    // supported resolutions run to run).
                    //
                    // Built via ResolutionSelector, not the deprecated
                    // setTargetResolution - real testing showed
                    // setTargetResolution had stopped reliably constraining
                    // anything at all on this device/CameraX 1.6.1 (asked
                    // for 400x300, got back a completely unrelated 1080x1080
                    // square) - it's a weak hint CameraX was free to ignore.
                    // ResolutionSelector + ResolutionStrategy is CameraX's
                    // actual current API for this and gives a firm
                    // bounding-size preference with an explicit fallback
                    // rule instead.
                    val analysisWidth = ANALYSIS_TARGET_WIDTH
                    val analysisHeight = (analysisWidth * 3 / 4)
                    val targetResolution = android.util.Size(analysisWidth, analysisHeight)
                    val resolutionSelector = ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(targetResolution, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                        )
                        .build()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    // STRATEGY_KEEP_ONLY_LATEST - if detection takes longer than a
                    // frame interval, drop stale frames rather than queueing them;
                    // we only ever care about the most recent one.
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setResolutionSelector(resolutionSelector)
                        .build()

                    // Bound once here alongside Preview/ImageAnalysis, never
                    // rebound later when a capture session starts/stops -
                    // rebinding use cases was exactly what made resolution
                    // negotiation flaky before (see md/status-2026-08-14-part2.md,
                    // Mistake 3). Only the aspect ratio is pinned (matching
                    // ImageAnalysis's forced 4:3) so the capture stream's crop
                    // matches the analysis stream's - resolution itself is left
                    // unconstrained so CameraX picks the sensor's largest
                    // still-capture size.
                    val imageCapture = ImageCapture.Builder()
                        .setResolutionSelector(
                            ResolutionSelector.Builder()
                                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                                .build()
                        )
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor) { image ->
                        try {
                            val mat = image.yuvToRgba()
                            val matSize = mat.size()
                            // preview.resolutionInfo is what Preview ACTUALLY
                            // negotiated - the one thing we've been guessing
                            // about. If this doesn't match "actual mat" below,
                            // the two streams are showing genuinely different
                            // crops of the sensor, which would explain a
                            // mismatch no amount of overlay-math tweaking can
                            // fix on its own.
                            val previewRes = preview.resolutionInfo?.resolution
                            debugFrameInfo = "requested ${analysisWidth}x${analysisHeight} | " +
                                "actual mat ${matSize.width.toInt()}x${matSize.height.toInt()} | " +
                                "preview res ${previewRes?.width}x${previewRes?.height} | " +
                                "rotation ${image.imageInfo.rotationDegrees} | " +
                                "view ${previewView.width}x${previewView.height}"
                            // Reads the toggles' LATEST values - each is a
                            // delegated property over a remembered State, so
                            // referencing it re-reads the current value at
                            // this call site, not a stale lambda-creation-time
                            // snapshot.
                            val guardConfig = GuardConfig(
                                edgeMarginGuardEnabled = edgeMarginGuardEnabled,
                                oppositeSideGuardEnabled = oppositeSideGuardEnabled,
                                minAreaGuardEnabled = minAreaGuardEnabled,
                                aspectRatioGuardEnabled = aspectRatioGuardEnabled,
                                centerPriorityEnabled = centerPriorityEnabled
                            )
                            val quad = nativeBridge.detectLargestQuadrilateral(mat, guardConfig)

                            val resultPoints = if (stabilizerEnabled) {
                                cornerStabilizer.update(quad?.points, matSize.width)
                            } else {
                                // Not just skipped - actively cleared, so
                                // toggling back on doesn't instantly average
                                // in stale samples from before it was off.
                                cornerStabilizer.reset()
                                quad?.points
                            }

                            // Sampled BEFORE releasing mat, from the same
                            // resultPoints that get displayed - both the
                            // live readout and anything collected into
                            // collectedSamples reflect the actual detected
                            // region, not a fluke frame that never shows.
                            val colorHere = resultPoints?.let { meanHsvInRegion(mat, it) }
                            mat.release()
                            sampledColor = colorHere

                            if (colorSamplingActive && colorHere != null) {
                                val updated = collectedSamples + colorHere
                                collectedSamples = updated
                                if (updated.size >= COLOR_SAMPLE_TARGET_COUNT) {
                                    colorSamplingActive = false
                                }
                            }

                            detectedCorners = resultPoints?.map { point ->
                                mapAnalysisPointToScreen(point, matSize.width, matSize.height, previewView.width, previewView.height)
                            }

                            // Guided capture: only fires while a session is
                            // active, on the same stabilizer-backed
                            // resultPoints used for display (several
                            // consecutive close frames, i.e. the user has
                            // actually stopped moving - forced via
                            // stabilizerEnabled=true when a session starts).
                            // Finding a new-cell pose doesn't capture
                            // immediately - it starts a HOLD_STILL_MS
                            // countdown (reset if the pose jumps to a
                            // different new cell, or tracking is lost) so
                            // the user's hand has time to move out of frame
                            // before the shutter actually fires.
                            val session = captureSession
                            if (resultPoints == null) {
                                pendingCell = null
                                holdStillCountdown = null
                            } else {
                                val signature = computeOrientation(resultPoints)
                                val cell = captureCoverage.cellFor(signature)
                                if (!sessionActive) {
                                    pendingCell = null
                                    holdStillCountdown = null
                                } else if (
                                    session != null && !captureCoverage.isComplete && !isCapturing &&
                                    System.currentTimeMillis() - lastCaptureTimeMs >= CAPTURE_COOLDOWN_MS
                                ) {
                                    val colorOk = if (colorGuardEnabled && lockedColor != null) {
                                        colorHere != null && matchesLockedColor(colorHere, lockedColor!!)
                                    } else {
                                        true
                                    }
                                    // A cell already has one capture - the
                                    // second (presumed 180-degree twin) is
                                    // only allowed if the immediately
                                    // preceding capture was of a DIFFERENT
                                    // cell, i.e. the two can't be consecutive.
                                    val blockedAsRepeat = captureCoverage.fillCountFor(signature) > 0 && lastCapturedCell == cell
                                    if (!colorOk) {
                                        pendingCell = null
                                        holdStillCountdown = null
                                    } else if (!captureCoverage.canCapture(signature) || blockedAsRepeat) {
                                        pendingCell = null
                                        holdStillCountdown = null
                                        sessionPrompt = "Already have this angle (${captureCoverage.filledCount}/20) - turn or tilt a bit more"
                                    } else {
                                        val now = System.currentTimeMillis()
                                        if (pendingCell != cell) {
                                            pendingCell = cell
                                            pendingSince = now
                                        }
                                        val elapsed = now - pendingSince
                                        if (elapsed < HOLD_STILL_MS) {
                                            holdStillCountdown = ceil((HOLD_STILL_MS - elapsed) / 1000.0).toInt()
                                            sessionPrompt = "New angle found - hold still"
                                        } else {
                                            pendingCell = null
                                            holdStillCountdown = null
                                            captureCoverage.reserve(signature)
                                            // Restored on failure below, so a
                                            // retry isn't wrongly blocked as
                                            // "consecutive" against a capture
                                            // that never actually happened.
                                            val previousLastCapturedCell = lastCapturedCell
                                            lastCapturedCell = cell
                                            isCapturing = true
                                            sessionPrompt = "Capturing..."
                                            val capturedMatWidth = matSize.width
                                            val capturedMatHeight = matSize.height
                                            val capturedPoints = resultPoints
                                            imageCapture.takePicture(
                                                cameraExecutor,
                                                object : ImageCapture.OnImageCapturedCallback() {
                                                    override fun onCaptureSuccess(image: ImageProxy) {
                                                        try {
                                                            saveCapturedFrame(
                                                                image,
                                                                capturedPoints,
                                                                capturedMatWidth,
                                                                capturedMatHeight,
                                                                signature,
                                                                session
                                                            )
                                                            coverageCount = captureCoverage.filledCount
                                                            sessionPrompt = if (captureCoverage.isComplete) {
                                                                "Coverage complete! Tap Stop to finish."
                                                            } else {
                                                                "Captured $coverageCount/20 - keep turning/tilting"
                                                            }
                                                        } catch (e: Exception) {
                                                            captureCoverage.release(signature)
                                                            lastCapturedCell = previousLastCapturedCell
                                                            sessionPrompt = "Capture failed, try again"
                                                            Log.e(CAPTURE_LOG_TAG, "Failed to save capture", e)
                                                        } finally {
                                                            image.close()
                                                            isCapturing = false
                                                            lastCaptureTimeMs = System.currentTimeMillis()
                                                        }
                                                    }

                                                    override fun onError(exception: ImageCaptureException) {
                                                        captureCoverage.release(signature)
                                                        lastCapturedCell = previousLastCapturedCell
                                                        isCapturing = false
                                                        sessionPrompt = "Capture failed, try again"
                                                        Log.e(CAPTURE_LOG_TAG, "takePicture failed", exception)
                                                    }
                                                }
                                            )
                                        }
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            detectedCorners = null
                            sampledColor = null
                        } finally {
                            image.close()
                        }
                    }

                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis, imageCapture
                    )
                    // Re-binding (e.g. after a config change) creates a new
                    // Camera instance with torch reset to off - reapply
                    // whatever the toggle's current state is rather than
                    // silently losing it.
                    if (camera.cameraInfo.hasFlashUnit()) {
                        camera.cameraControl.enableTorch(flashEnabled)
                    }
                    boundCamera = camera
                }, ContextCompat.getMainExecutor(context))
            }

            onDispose {
                if (cameraProviderFuture.isDone) {
                    cameraProviderFuture.get().unbindAll()
                }
                cameraExecutor.shutdown()
            }
        }

        val colorMatches = if (colorGuardEnabled && lockedColor != null) {
            sampledColor?.let { matchesLockedColor(it, lockedColor!!) } ?: false
        } else {
            true
        }
        val matched = colorMatches

        Box(modifier = Modifier.fillMaxWidth().height(300.dp)) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            DetectionOverlay(detectedCorners, matched = matched, modifier = Modifier.fillMaxSize())
            // Large, high-contrast countdown drawn directly over the live
            // feed - the point is to be impossible to miss while the user's
            // hand is still near the foam, unlike the small status text
            // below the preview.
            holdStillCountdown?.let { seconds ->
                Text(
                    "HOLD STILL\n$seconds",
                    fontSize = 40.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .align(Alignment.Center)
                        .background(Color(0xAA000000))
                        .padding(24.dp)
                )
            }
        }

        val (statusText, statusColor) = when {
            detectedCorners == null -> "Not detected" to Color(0xFFFF4444)
            !colorMatches -> "Shape found - wrong color" to Color(0xFFFFAA00)
            else -> "Detected" to Color(0xFF33CC33)
        }
        Text(statusText, fontSize = 18.sp, color = statusColor)
        if (debugFrameInfo.isNotEmpty()) {
            Text(debugFrameInfo, fontSize = 12.sp)
        }

        // Live readout - point the camera at the real object and watch this
        // to judge whether Sample/Lock is capturing something sensible.
        sampledColor?.let {
            Text("Live H=${it.hue.roundToInt()} S=${it.saturation.roundToInt()} V=${it.value.roundToInt()}", fontSize = 14.sp)
        }
        lockedColor?.let {
            Text(
                "Locked H=${it.hue.roundToInt()} S=${it.saturation.roundToInt()} V=${it.value.roundToInt()}",
                fontSize = 14.sp
            )
        }

        if (sessionActive) {
            // Replaces the tuning/guard controls entirely while a session
            // runs - progress + the current guidance prompt, plus Stop,
            // which works at any point regardless of how many angles have
            // been captured so far.
            Text("Captured $coverageCount/20 unique angles", fontSize = 16.sp)
            if (sessionPrompt.isNotEmpty()) {
                Text(sessionPrompt, fontSize = 14.sp)
            }
            Button(
                onClick = {
                    sessionActive = false
                    lastCapturedCell = null
                    pendingCell = null
                    holdStillCountdown = null
                },
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            ) {
                Text("Stop scan")
            }
        } else {
            Button(
                onClick = {
                    captureCoverage.reset()
                    coverageCount = 0
                    stabilizerEnabled = true
                    captureSession = CaptureSession(context)
                    sessionPrompt = "Show the foam to begin"
                    lastCapturedCell = null
                    pendingCell = null
                    holdStillCountdown = null
                    sessionActive = true
                },
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            ) {
                Text("Start scan (20 angles)")
            }
        }

        // 2-per-row grid instead of one button per row - keeps the whole
        // control area compact and predictable regardless of how many
        // toggles exist, rather than a single tall column. Hidden during a
        // capture session so it doesn't compete with the guided-scan UI.
        val toggleButtons = if (sessionActive) emptyList() else listOf<Pair<String, () -> Unit>>(
            (
                if (colorSamplingActive) {
                    "Sampling... (${collectedSamples.size}/$COLOR_SAMPLE_TARGET_COUNT)"
                } else {
                    "Sample color (${collectedSamples.size})"
                }
                ) to {
                if (colorSamplingActive) {
                    colorSamplingActive = false
                } else {
                    collectedSamples = emptyList()
                    colorSamplingActive = true
                }
            },
            (if (lockedColor != null) "Re-lock color sample" else "Lock color sample") to {
                if (collectedSamples.isNotEmpty()) {
                    lockedColor = averageColor(collectedSamples)
                    colorGuardEnabled = true
                }
            },
            (if (colorGuardEnabled) "Color guard: ON" else "Color guard: OFF") to {
                colorGuardEnabled = !colorGuardEnabled
            },
            (if (flashEnabled) "Flashlight: ON" else "Flashlight: OFF") to {
                val camera = boundCamera
                if (camera != null && camera.cameraInfo.hasFlashUnit()) {
                    flashEnabled = !flashEnabled
                    camera.cameraControl.enableTorch(flashEnabled)
                }
            },
            (if (stabilizerEnabled) "Corner stabilizer: ON" else "Corner stabilizer: OFF") to {
                stabilizerEnabled = !stabilizerEnabled
            },
            (if (edgeMarginGuardEnabled) "Edge-margin guard: ON" else "Edge-margin guard: OFF") to {
                edgeMarginGuardEnabled = !edgeMarginGuardEnabled
            },
            (if (oppositeSideGuardEnabled) "Opposite-side guard: ON" else "Opposite-side guard: OFF") to {
                oppositeSideGuardEnabled = !oppositeSideGuardEnabled
            },
            (if (minAreaGuardEnabled) "Min-area guard: ON" else "Min-area guard: OFF") to {
                minAreaGuardEnabled = !minAreaGuardEnabled
            },
            (if (aspectRatioGuardEnabled) "Aspect-ratio guard: ON" else "Aspect-ratio guard: OFF") to {
                aspectRatioGuardEnabled = !aspectRatioGuardEnabled
            },
            (if (centerPriorityEnabled) "Center-priority: ON" else "Center-priority: OFF") to {
                centerPriorityEnabled = !centerPriorityEnabled
            }
        )
        toggleButtons.chunked(2).forEach { row ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                row.forEach { (label, onClick) ->
                    Button(onClick = onClick, modifier = Modifier.weight(1f)) {
                        Text(label, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

// Analysis frames come from the sensor in its native (landscape) orientation
// regardless of how the phone is held, while the preview is displayed in
// portrait - this reprojects a raw OpenCV point (in un-rotated analysis-Mat
// pixel space) into the portrait view's own pixel coordinates. Same fixed
// 90-degree-rotation assumption as the source SDK's ScanCanvasView.showShape,
// which only targets portrait use - fine for our case, not a general solution.
//
// Real measured numbers (analysis Mat 640x480, PreviewView 996x960) showed
// the previous version was wrong beyond just the rotation: it stretched the
// content independently on X and Y to exactly fill the view, but
// PreviewView's actual default ScaleType is FILL_CENTER - a single UNIFORM
// scale that covers the view and crops whatever overflows, centered. Those
// are genuinely different transforms whenever the content and view aspect
// ratios don't match (here: content 480x640 after rotation vs. a nearly
// square 996x960 view) - the mismatch was exactly why the green outline
// didn't reach the real top/bottom edges of a detected rectangle: the
// overlay was mapped as if the full sensor image were visible uncropped,
// when PreviewView was actually cropping its top and bottom to fill the
// wider container.
private fun mapAnalysisPointToScreen(
    point: Point,
    matWidth: Double,
    matHeight: Double,
    viewWidthPx: Int,
    viewHeightPx: Int
): Offset {
    if (viewWidthPx == 0 || viewHeightPx == 0) return Offset.Zero
    // Content dimensions after the fixed 90-degree rotation, before any
    // view-fitting scale/crop.
    val contentWidth = matHeight.toFloat()
    val contentHeight = matWidth.toFloat()
    val logicalX = contentWidth - point.y.toFloat()
    val logicalY = point.x.toFloat()

    // FILL_CENTER: one scale factor for both axes (the larger of the two
    // "fill this dimension" ratios, so the content covers the view
    // completely), then centered - matching PreviewView's own rendering
    // instead of an independent-axis stretch.
    val scale = max(viewWidthPx / contentWidth, viewHeightPx / contentHeight)
    val offsetX = (viewWidthPx - contentWidth * scale) / 2f
    val offsetY = (viewHeightPx - contentHeight * scale) / 2f

    return Offset(
        offsetX + logicalX * scale,
        offsetY + logicalY * scale
    )
}

// Saves TWO files per capture (see CaptureStorage.kt's class doc for why):
// the raw captured JPEG bytes untouched (preserves EXIF - the intended
// photogrammetry input), and a crop of it down to the detected quad
// (a convenience copy only, not meant to feed reconstruction). The crop is
// deliberately a plain crop, not a perspective warp - the perspective
// distortion is exactly the signal a backend photogrammetry/3D step needs
// from each of the ~20 angles; warping it away would erase what makes the
// shots different viewpoints.
//
// image's JPEG bytes come in the sensor's native (un-rotated) orientation -
// the same convention resultPoints/matWidth/matHeight already use - so the
// quad corners are mapped into that native pixel space with a pure scale,
// no rotation. Only the small crop gets rotated (by
// imageInfo.rotationDegrees) into an upright, human-viewable orientation -
// the raw full-size bytes are saved completely as-is, no manual rotation
// applied, so they can't pick up a double-rotation bug if the camera's own
// EXIF orientation tag already accounts for it. This orientation assumption
// is exactly the kind that has broken this project before (see
// md/status-2026-08-14-part2.md) - logged here so it's verifiable against a
// real device rather than trusted blindly.
private fun saveCapturedFrame(
    image: ImageProxy,
    matPoints: Array<Point>,
    matWidth: Double,
    matHeight: Double,
    signature: OrientationSignature,
    session: CaptureSession
) {
    val buffer = image.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    val nativeBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        ?: throw IllegalStateException("Failed to decode captured JPEG")

    val rotationDegrees = image.imageInfo.rotationDegrees

    Log.d(
        CAPTURE_LOG_TAG,
        "native ${nativeBitmap.width}x${nativeBitmap.height} | mat ${matWidth.toInt()}x${matHeight.toInt()} | " +
            "rotation $rotationDegrees"
    )

    val scaleX = nativeBitmap.width / matWidth
    val scaleY = nativeBitmap.height / matHeight
    val xs = matPoints.map { it.x * scaleX }
    val ys = matPoints.map { it.y * scaleY }
    val minX = xs.min()
    val maxX = xs.max()
    val minY = ys.min()
    val maxY = ys.max()
    val marginX = (maxX - minX) * CROP_MARGIN_RATIO
    val marginY = (maxY - minY) * CROP_MARGIN_RATIO

    val left = (minX - marginX).toInt().coerceIn(0, nativeBitmap.width - 1)
    val top = (minY - marginY).toInt().coerceIn(0, nativeBitmap.height - 1)
    val right = (maxX + marginX).toInt().coerceIn(left + 1, nativeBitmap.width)
    val bottom = (maxY + marginY).toInt().coerceIn(top + 1, nativeBitmap.height)
    val cropRect = Rect(left, top, right, bottom)

    val cropped = Bitmap.createBitmap(nativeBitmap, left, top, right - left, bottom - top)
    val finalBitmap = if (rotationDegrees != 0) {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
    } else {
        cropped
    }

    val nativeCorners = matPoints.mapIndexed { i, _ -> Point(xs[i], ys[i]) }.toTypedArray()

    session.saveCapture(
        fullJpegBytes = bytes,
        croppedBitmap = finalBitmap,
        signature = signature,
        nativeImageWidth = nativeBitmap.width,
        nativeImageHeight = nativeBitmap.height,
        cropRectNative = cropRect,
        cornersNative = nativeCorners,
        rotationDegreesApplied = rotationDegrees
    )
}

@Composable
private fun DetectionOverlay(corners: List<Offset>?, matched: Boolean, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        if (corners == null || corners.size != 4) return@Canvas

        // Green once the color check also confirms it (or the color guard
        // is off/not yet locked), amber if a shape was found but the locked
        // color check rejected it.
        val outlineColor = if (matched) Color(0xFF33CC33) else Color(0xFFFFAA00)
        val strokeWidth = 4.dp.toPx()
        for (i in corners.indices) {
            val start = corners[i]
            val end = corners[(i + 1) % corners.size]
            drawLine(color = outlineColor, start = start, end = end, strokeWidth = strokeWidth)
        }
        for (corner in corners) {
            drawCircle(color = outlineColor, radius = 10.dp.toPx(), center = corner, style = Stroke(width = strokeWidth))
        }
    }
}
