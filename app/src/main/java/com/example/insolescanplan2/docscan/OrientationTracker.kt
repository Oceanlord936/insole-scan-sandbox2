package com.example.insolescanplan2.docscan

import org.opencv.core.Point
import kotlin.math.atan2

// A rectangle looks identical after a 180-degree in-plane rotation - a
// single frame's quad shape genuinely cannot tell 30 degrees from 210
// degrees apart, the 4 corner positions are pixel-identical either way. The
// tilt ratios have the same limitation as a bonus - shape alone just isn't
// enough.
//
// Frame-to-frame motion tracking can't resolve this either, despite being
// the obvious next idea: the foam can't be turned without a hand covering
// part of it first (there's no way to reposition it otherwise), so every
// transition between held poses is visible -> occluded (detection fails,
// the hand breaks the contour) -> visible again, never a continuous path of
// small angle deltas to integrate. See CaptureCoverage below for how this
// gets handled instead - via the occlusion itself, not by trying to track
// through it.
internal data class OrientationSignature(
    val angleDeg: Double, // mod 180 - see above
    val topBottomTilt: Double,
    val leftRightTilt: Double
)

// points follows OpenCvNativeBridge.sortPoints' convention:
// [0]=top-left, [1]=top-right, [2]=bottom-right, [3]=bottom-left.
internal fun computeOrientation(points: Array<Point>): OrientationSignature {
    val (tl, tr, br, bl) = points

    val top = edgeLength(tl, tr)
    val right = edgeLength(tr, br)
    val bottom = edgeLength(br, bl)
    val left = edgeLength(bl, tl)

    // Long axis = average direction of whichever pair of opposite edges is
    // longer, summing both vectors rather than just picking one side so a
    // single noisy corner doesn't dominate the angle estimate.
    val avgTopBottom = (top + bottom) / 2.0
    val avgLeftRight = (left + right) / 2.0
    val axisX: Double
    val axisY: Double
    if (avgTopBottom >= avgLeftRight) {
        axisX = (tr.x - tl.x) + (br.x - bl.x)
        axisY = (tr.y - tl.y) + (br.y - bl.y)
    } else {
        axisX = (bl.x - tl.x) + (br.x - tr.x)
        axisY = (bl.y - tl.y) + (br.y - tr.y)
    }
    val rawDeg = Math.toDegrees(atan2(axisY, axisX))
    // Wrap into [0, 180) - a line has no distinguishable direction beyond
    // that.
    val angleDeg = ((rawDeg % 180.0) + 180.0) % 180.0

    val topBottomTilt = if (top + bottom > 0.0) (top - bottom) / (top + bottom) else 0.0
    val leftRightTilt = if (left + right > 0.0) (left - right) / (left + right) else 0.0

    return OrientationSignature(angleDeg, topBottomTilt, leftRightTilt)
}

private fun edgeLength(a: Point, b: Point): Double {
    val dx = a.x - b.x
    val dy = a.y - b.y
    return Math.sqrt(dx * dx + dy * dy)
}

// Buckets an OrientationSignature into a coarse cell and tracks how many
// times each has been captured, instead of comparing each new frame against
// every past sample - gives an exact, deterministic coverage count ("14/20")
// and a clean stop condition, rather than guessing completion from a streak
// of repeat detections.
//
// Each cell allows up to 2 captures rather than 1: one for "this shape", and
// one more for its 180-degree-apart twin, which is genuinely indistinguishable
// from the first by shape alone. DocScanScreen only allows that second
// capture if the immediately preceding capture was of a DIFFERENT cell - i.e.
// the two captures of the same cell can't be consecutive. That's stricter
// than just requiring an occlusion in between (a nervous re-grab that lands
// back on the same pose would still count as "occluded"), and it still
// implies real repositioning happened, since reaching a different cell isn't
// possible without the hand covering the object in between.
internal class CaptureCoverage(private val targetCells: Int = 20, private val maxFillsPerCell: Int = 2) {
    companion object {
        // 180 degrees / 10 bins = 18 degrees per bin.
        private const val ANGLE_BIN_COUNT = 10
        // Same starting-guess-then-tune constant style as ColorCheck.kt's
        // tolerances - a tilt ratio inside this band counts as "flat" on
        // that axis, outside it counts as tilted one way or the other.
        private const val TILT_NEUTRAL_THRESHOLD = 0.08
    }

    private val fillCounts = mutableMapOf<Triple<Int, Int, Int>, Int>()

    val filledCount: Int get() = fillCounts.values.sum()
    val isComplete: Boolean get() = filledCount >= targetCells

    fun fillCountFor(signature: OrientationSignature): Int = fillCounts[cellFor(signature)] ?: 0

    fun canCapture(signature: OrientationSignature): Boolean = fillCountFor(signature) < maxFillsPerCell

    fun reserve(signature: OrientationSignature) {
        val cell = cellFor(signature)
        fillCounts[cell] = (fillCounts[cell] ?: 0) + 1
    }

    fun release(signature: OrientationSignature) {
        val cell = cellFor(signature)
        val current = fillCounts[cell] ?: return
        if (current <= 1) fillCounts.remove(cell) else fillCounts[cell] = current - 1
    }

    fun reset() {
        fillCounts.clear()
    }

    // Exposed (not private) so DocScanScreen can identify whether the
    // pose currently being held is the same cell it started a hold-still
    // countdown for, or a different one that should restart it.
    fun cellFor(signature: OrientationSignature): Triple<Int, Int, Int> {
        val angleBin = (signature.angleDeg / 180.0 * ANGLE_BIN_COUNT)
            .toInt()
            .coerceIn(0, ANGLE_BIN_COUNT - 1)
        return Triple(angleBin, tiltBin(signature.topBottomTilt), tiltBin(signature.leftRightTilt))
    }

    private fun tiltBin(tilt: Double): Int = when {
        tilt < -TILT_NEUTRAL_THRESHOLD -> -1
        tilt > TILT_NEUTRAL_THRESHOLD -> 1
        else -> 0
    }
}
