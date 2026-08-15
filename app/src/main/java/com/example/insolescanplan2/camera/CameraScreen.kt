package com.example.insolescanplan2.camera

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner

// Bare-minimum CameraX screen for plan 2 (see md/08-12-2026-plan-cv-rotation-scan.md
// in InsoleScanSandbox): live preview + a capture button, no Hilt/ViewModel/Room -
// this project starts from just enough scaffolding to point a live feed at the
// foam and confirm OpenCV can find its rotated rectangle, per the plan's own
// recommended first step. Status is plain local Compose state for now.
@Composable
fun CameraScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember { PreviewView(context) }
    var statusText by remember { mutableStateOf("No photo yet") }

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    // Tied to this screen's lifetime via remember, not a ViewModel - it wraps
    // Context/lifecycle-bound camera resources.
    val imageCapture = remember { ImageCapture.Builder().build() }

    Column(modifier = modifier.padding(16.dp)) {
        if (!hasPermission) {
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Grant camera permission")
            }
            return@Column
        }

        // Bind Preview + ImageCapture once permission exists - DisposableEffect
        // (not LaunchedEffect) because binding a camera is a resource that needs
        // an explicit matching unbind, not just a coroutine to cancel.
        DisposableEffect(lifecycleOwner) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            }, ContextCompat.getMainExecutor(context))

            onDispose {
                // Guard against .get() blocking on a future that hasn't resolved
                // yet if this composable is torn down almost immediately after binding.
                if (cameraProviderFuture.isDone) {
                    cameraProviderFuture.get().unbindAll()
                }
            }
        }

        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxWidth().height(400.dp)
        )

        Button(onClick = {
            // MediaStore.MediaColumns.RELATIVE_PATH only exists on API 29+ (scoped
            // storage) - inserting with it below API 29 can throw IllegalArgumentException.
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                statusText = "Saving photos requires Android 10 (API 29) or higher"
                return@Button
            }

            val name = "InsoleScanPlan2_${System.currentTimeMillis()}.jpg"
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/InsoleScanPlan2")
            }
            val outputOptions = ImageCapture.OutputFileOptions.Builder(
                context.contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ).build()

            imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        statusText = "Saved: ${output.savedUri}"
                    }

                    override fun onError(exception: ImageCaptureException) {
                        statusText = "Error: ${exception.message}"
                    }
                }
            )
        }) {
            Text("Capture Photo")
        }

        Text(statusText)
    }
}
