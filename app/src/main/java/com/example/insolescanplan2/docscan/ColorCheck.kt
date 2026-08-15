package com.example.insolescanplan2.docscan

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

data class SampledColor(val hue: Double, val saturation: Double, val value: Double)

// Tolerance band applied around whatever color gets locked via the
// Sample/Lock workflow in DocScanScreen - a fixed buffer around the
// measured average, not derived from the sample spread itself. Same
// starting-guess-then-recalibrate pattern as every other constant in this
// project: reasonable default, meant to be tightened/loosened once you can
// see how strict/loose it feels against the real object.
private const val HUE_TOLERANCE = 15.0
private const val SATURATION_TOLERANCE = 50.0
private const val VALUE_TOLERANCE = 50.0

// Confirms a geometrically-detected quadrilateral is actually the expected
// object by color, using a color CALIBRATED live from the real object
// (Sample button collects several readings while pointed at it, Lock
// button averages them into the reference) instead of a guessed HSV range -
// this runs strictly AFTER geometric detection, as a confirmation gate on
// the region the shape detector already found, not a detector itself.
fun meanHsvInRegion(src: Mat, points: Array<Point>): SampledColor {
    val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
    Imgproc.fillConvexPoly(mask, MatOfPoint(*points), Scalar(255.0))
    val meanColor = Core.mean(src, mask)
    mask.release()

    // src is nominally RGBA (see ImageProxyExtensions.yuvToRgba - one of its
    // two conversion paths actually produces BGRA, an inconsistency
    // inherited from the source SDK) - exact channel order doesn't matter
    // here as long as sampling and locking both go through this same
    // function, which they do.
    val sampleMat = Mat(1, 1, CvType.CV_8UC3)
    sampleMat.put(
        0, 0,
        byteArrayOf(meanColor.`val`[0].toInt().toByte(), meanColor.`val`[1].toInt().toByte(), meanColor.`val`[2].toInt().toByte())
    )
    val hsvMat = Mat()
    Imgproc.cvtColor(sampleMat, hsvMat, Imgproc.COLOR_RGB2HSV)
    val hsvBytes = ByteArray(3)
    hsvMat.get(0, 0, hsvBytes)
    hsvMat.release()
    sampleMat.release()

    return SampledColor(
        hue = (hsvBytes[0].toInt() and 0xFF).toDouble(),
        saturation = (hsvBytes[1].toInt() and 0xFF).toDouble(),
        value = (hsvBytes[2].toInt() and 0xFF).toDouble()
    )
}

fun averageColor(samples: List<SampledColor>): SampledColor {
    return SampledColor(
        hue = samples.sumOf { it.hue } / samples.size,
        saturation = samples.sumOf { it.saturation } / samples.size,
        value = samples.sumOf { it.value } / samples.size
    )
}

fun matchesLockedColor(color: SampledColor, locked: SampledColor): Boolean {
    return abs(color.hue - locked.hue) <= HUE_TOLERANCE &&
        abs(color.saturation - locked.saturation) <= SATURATION_TOLERANCE &&
        abs(color.value - locked.value) <= VALUE_TOLERANCE
}
