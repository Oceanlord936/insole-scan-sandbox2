package com.example.insolescanplan2.docscan

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Point
import java.io.File

// One capture session's worth of saved images + metadata.
//
// Images go through MediaStore into the public Pictures collection (same
// approach CameraScreen.kt's dead-code capture button already used for a
// single photo, extended here to a per-session subfolder) so they actually
// show up in the phone's Gallery app - app-specific external storage
// (getExternalFilesDir, this class's previous approach) is invisible to
// Gallery/Files apps entirely, which is why nothing appeared there.
// MediaStore.Images insert works without any runtime permission on API 29+
// (scoped storage); below that, saving to the public gallery needs
// WRITE_EXTERNAL_STORAGE and a legacy direct-file-path insert this project
// hasn't needed yet, so - matching CameraScreen.kt's own existing choice -
// that's surfaced as a clear failure rather than silently reimplemented.
//
// The manifest stays in app-specific storage: it's not a photo, nobody
// expects to find a JSON file in their gallery, and keeping it off
// MediaStore avoids needing a second, non-image collection (Downloads) just
// for one small sidecar file. It's kept in native (un-rotated, pre-crop)
// capture-image pixel space rather than pre-transformed into "final saved
// image" space, so nothing here has to assume its own rotation/crop math is
// correct - a backend consumer can redo that transform from the raw numbers
// if needed.
internal class CaptureSession(private val context: Context) {
    private val sessionId = System.currentTimeMillis()
    private val galleryRelativeDir = "Pictures/InsoleScan/session_$sessionId"
    private val manifestDir: File = File(
        context.getExternalFilesDir(null),
        "scan_sessions/session_$sessionId"
    ).apply { mkdirs() }

    private val manifestEntries = JSONArray()
    private var nextIndex = 1

    fun saveCapture(
        finalBitmap: Bitmap,
        signature: OrientationSignature,
        nativeImageWidth: Int,
        nativeImageHeight: Int,
        cropRectNative: Rect,
        cornersNative: Array<Point>,
        rotationDegreesApplied: Int
    ): Uri {
        val filename = "image_%03d.jpg".format(nextIndex++)
        val uri = saveToGallery(filename, finalBitmap)

        val entry = JSONObject().apply {
            put("file", filename)
            put("uri", uri.toString())
            put("timestampMs", System.currentTimeMillis())
            put("angleDeg", signature.angleDeg)
            put("topBottomTilt", signature.topBottomTilt)
            put("leftRightTilt", signature.leftRightTilt)
            put("nativeImageWidth", nativeImageWidth)
            put("nativeImageHeight", nativeImageHeight)
            put(
                "cropRectNative",
                JSONObject().apply {
                    put("left", cropRectNative.left)
                    put("top", cropRectNative.top)
                    put("right", cropRectNative.right)
                    put("bottom", cropRectNative.bottom)
                }
            )
            put(
                "cornersNative",
                JSONArray().apply {
                    cornersNative.forEach { p ->
                        put(JSONArray().apply { put(p.x); put(p.y) })
                    }
                }
            )
            put("rotationDegreesApplied", rotationDegreesApplied)
        }
        manifestEntries.put(entry)
        writeManifest()
        return uri
    }

    private fun saveToGallery(filename: String, bitmap: Bitmap): Uri {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            throw IllegalStateException("Saving to the gallery requires Android 10 (API 29) or higher")
        }
        val resolver = context.contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, galleryRelativeDir)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            ?: throw IllegalStateException("MediaStore insert failed for $filename")

        resolver.openOutputStream(uri)?.use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        } ?: throw IllegalStateException("Failed to open output stream for $uri")

        contentValues.clear()
        contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
        resolver.update(uri, contentValues, null, null)

        return uri
    }

    private fun writeManifest() {
        val manifest = JSONObject().apply {
            put("images", manifestEntries)
        }
        File(manifestDir, "manifest.json").writeText(manifest.toString(2))
    }
}
