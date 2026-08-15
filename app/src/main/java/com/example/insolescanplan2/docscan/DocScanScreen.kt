package com.example.insolescanplan2.docscan

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt

// Matches the source SDK's own IMAGE_ANALYSIS_SCALE_WIDTH - the detection
// algorithm's fixed-pixel kernel sizes were tuned against frames this wide.
// Not an algorithm change - matching the original's own assumption.
private const val ANALYSIS_TARGET_WIDTH = 400

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

    val previewView = remember { PreviewView(context) }
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

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    Column(modifier = modifier.padding(16.dp)) {
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
            // Wait for previewView to actually be laid out (like the source SDK's
            // viewFinder.post { ... }) - we need its real aspect ratio below, and
            // width/height read 0 before the first layout pass.
            previewView.post {
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val previewWidth = previewView.width.takeIf { it > 0 } ?: 1080
                    val previewHeight = previewView.height.takeIf { it > 0 } ?: 1920
                    val aspectRatio = previewWidth.toFloat() / previewHeight.toFloat()
                    val analysisWidth = ANALYSIS_TARGET_WIDTH
                    val analysisHeight = (analysisWidth / aspectRatio).roundToInt()

                    // STRATEGY_KEEP_ONLY_LATEST - if detection takes longer than a
                    // frame interval, drop stale frames rather than queueing them;
                    // we only ever care about the most recent one.
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetResolution(android.util.Size(analysisWidth, analysisHeight))
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor) { image ->
                        try {
                            val mat = image.yuvToRgba()
                            val matSize = mat.size()
                            debugFrameInfo = "requested ${analysisWidth}x${analysisHeight} | " +
                                "actual mat ${matSize.width.toInt()}x${matSize.height.toInt()} | " +
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
                            mat.release()

                            val resultPoints = if (stabilizerEnabled) {
                                cornerStabilizer.update(quad?.points, matSize.width)
                            } else {
                                // Not just skipped - actively cleared, so
                                // toggling back on doesn't instantly average
                                // in stale samples from before it was off.
                                cornerStabilizer.reset()
                                quad?.points
                            }

                            detectedCorners = resultPoints?.map { point ->
                                mapAnalysisPointToScreen(point, matSize.width, matSize.height, previewView.width, previewView.height)
                            }
                        } catch (e: Exception) {
                            detectedCorners = null
                        } finally {
                            image.close()
                        }
                    }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
                    )
                }, ContextCompat.getMainExecutor(context))
            }

            onDispose {
                if (cameraProviderFuture.isDone) {
                    cameraProviderFuture.get().unbindAll()
                }
                cameraExecutor.shutdown()
            }
        }

        Box(modifier = Modifier.fillMaxWidth().weight(1f)) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            DetectionOverlay(detectedCorners, modifier = Modifier.fillMaxSize())
        }

        Text(
            if (detectedCorners != null) "Detected" else "Not detected",
            fontSize = 18.sp,
            color = if (detectedCorners != null) Color(0xFF33CC33) else Color(0xFFFF4444)
        )
        if (debugFrameInfo.isNotEmpty()) {
            Text(debugFrameInfo, fontSize = 12.sp)
        }

        Button(onClick = { stabilizerEnabled = !stabilizerEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (stabilizerEnabled) "Corner stabilizer: ON" else "Corner stabilizer: OFF")
        }
        Button(onClick = { edgeMarginGuardEnabled = !edgeMarginGuardEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (edgeMarginGuardEnabled) "Edge-margin guard: ON" else "Edge-margin guard: OFF")
        }
        Button(onClick = { oppositeSideGuardEnabled = !oppositeSideGuardEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (oppositeSideGuardEnabled) "Opposite-side guard: ON" else "Opposite-side guard: OFF")
        }
        Button(onClick = { minAreaGuardEnabled = !minAreaGuardEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (minAreaGuardEnabled) "Min-area guard: ON" else "Min-area guard: OFF")
        }
        Button(onClick = { aspectRatioGuardEnabled = !aspectRatioGuardEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (aspectRatioGuardEnabled) "Aspect-ratio guard: ON" else "Aspect-ratio guard: OFF")
        }
        Button(onClick = { centerPriorityEnabled = !centerPriorityEnabled }, modifier = Modifier.padding(top = 8.dp)) {
            Text(if (centerPriorityEnabled) "Center-priority: ON" else "Center-priority: OFF")
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

@Composable
private fun DetectionOverlay(corners: List<Offset>?, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        if (corners == null || corners.size != 4) return@Canvas

        val strokeWidth = 4.dp.toPx()
        for (i in corners.indices) {
            val start = corners[i]
            val end = corners[(i + 1) % corners.size]
            drawLine(color = Color(0xFF33CC33), start = start, end = end, strokeWidth = strokeWidth)
        }
        for (corner in corners) {
            drawCircle(color = Color(0xFF33CC33), radius = 10.dp.toPx(), center = corner, style = Stroke(width = strokeWidth))
        }
    }
}
