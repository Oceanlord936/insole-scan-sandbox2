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
import java.io.OutputStream

// One capture session's worth of saved images + metadata.
//
// Every capture saves TWO files, in separate subfolders:
// - "full": the original captured JPEG bytes, written untouched (no
//   decode/recompress round-trip) so EXIF (focal length, sensor size, the
//   camera's own orientation tag) survives intact - this is the copy meant
//   to feed a photogrammetry pipeline, which typically wants a consistent
//   camera model across all images and uses EXIF for initial intrinsics.
//   Also untouched by our own crop/rotate math, so it can't inherit a bug
//   from that.
// - "crop": the existing cropped-to-the-detected-quad, upright-rotated
//   bitmap - a convenience copy for on-device viewing/debugging, not meant
//   to be the reconstruction input (per-shot crop position/size varies,
//   which would give a photogrammetry tool an inconsistent principal point
//   per image if it were used as the primary source).
//
// Images go through MediaStore into the public Pictures collection (same
// approach CameraScreen.kt's dead-code capture button already used for a
// single photo) so they actually show up in the phone's Gallery app -
// app-specific external storage (getExternalFilesDir, this class's earlier
// approach) is invisible to Gallery/Files apps entirely, which is why
// nothing appeared there. MediaStore.Images insert works without any
// runtime permission on API 29+ (scoped storage); below that, saving to the
// public gallery needs WRITE_EXTERNAL_STORAGE and a legacy direct-file-path
// insert this project hasn't needed yet, so - matching CameraScreen.kt's
// own existing choice - that's surfaced as a clear failure rather than
// silently reimplemented.
//
// The manifest stays in app-specific storage: it's not a photo, nobody
// expects to find a JSON file in their gallery, and keeping it off
// MediaStore avoids needing a second, non-image collection (Downloads) just
// for one small sidecar file. Crop/corner metadata is kept in native
// (un-rotated, pre-crop) capture-image pixel space - the same space the
// "full" file is saved in - rather than pre-transformed into "final crop"
// space, so nothing here has to assume its own rotation/crop math is
// correct; a backend consumer can redo that transform from the raw numbers
// against the full file if needed.
internal class CaptureSession(private val context: Context) {
    private val sessionId = System.currentTimeMillis()
    private val fullRelativeDir = "Pictures/InsoleScan/session_$sessionId/full"
    private val cropRelativeDir = "Pictures/InsoleScan/session_$sessionId/crop"
    private val manifestDir: File = File(
        context.getExternalFilesDir(null),
        "scan_sessions/session_$sessionId"
    ).apply { mkdirs() }

    private val manifestEntries = JSONArray()
    private var nextIndex = 1

    fun saveCapture(
        fullJpegBytes: ByteArray,
        croppedBitmap: Bitmap,
        signature: OrientationSignature,
        nativeImageWidth: Int,
        nativeImageHeight: Int,
        cropRectNative: Rect,
        cornersNative: Array<Point>,
        rotationDegreesApplied: Int
    ) {
        val index = nextIndex++
        val fullFilename = "image_%03d.jpg".format(index)
        val cropFilename = "image_%03d.jpg".format(index)

        val fullUri = insertIntoGallery(fullFilename, fullRelativeDir) { out -> out.write(fullJpegBytes) }
        val cropUri = insertIntoGallery(cropFilename, cropRelativeDir) { out ->
            croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }

        val entry = JSONObject().apply {
            put("fullFile", "full/$fullFilename")
            put("fullUri", fullUri.toString())
            put("cropFile", "crop/$cropFilename")
            put("cropUri", cropUri.toString())
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
    }

    private fun insertIntoGallery(filename: String, relativeDir: String, write: (OutputStream) -> Unit): Uri {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            throw IllegalStateException("Saving to the gallery requires Android 10 (API 29) or higher")
        }
        val resolver = context.contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, relativeDir)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            ?: throw IllegalStateException("MediaStore insert failed for $filename")

        resolver.openOutputStream(uri)?.use(write)
            ?: throw IllegalStateException("Failed to open output stream for $uri")

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
