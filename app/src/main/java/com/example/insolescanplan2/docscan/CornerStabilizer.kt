package com.example.insolescanplan2.docscan

import org.opencv.core.Point
import kotlin.math.sqrt

// Raw per-frame detection has no memory of the previous frame, which is
// exactly why it looks like it's jumping everywhere: a single frame's noisy
// edge is shown immediately, with nothing to distinguish "found the real
// object again" from "found a one-frame fluke". This buffers recent raw
// detections and only reports a stable result once a few consecutive ones
// land close together (real object, held roughly still), discarding the
// buffer and starting over whenever a new detection jumps far from the last
// one (probably a different, unrelated candidate, not the same object
// having moved). Brief single-frame misses don't immediately clear the
// display either - holds the last stable result for a few frames first,
// since a dropped frame is common and shouldn't look identical to "object
// left the scene".
internal class CornerStabilizer(
    private val requiredSamples: Int = 3,
    // Ratio of frame width, not absolute pixels - scale-free, holds
    // regardless of analysis resolution.
    private val maxCornerDriftRatio: Double = 0.08,
    private val missToleranceFrames: Int = 5
) {
    private val recentSamples = mutableListOf<Array<Point>>()
    private var missStreak = 0
    private var stablePoints: Array<Point>? = null

    // Call once per frame with that frame's raw detection (or null if
    // nothing was detected) and the frame's width (for the drift
    // tolerance). Returns the current stable result, or null if nothing's
    // locked in yet.
    fun update(newPoints: Array<Point>?, frameWidth: Double): Array<Point>? {
        if (newPoints == null) {
            missStreak++
            if (missStreak > missToleranceFrames) {
                recentSamples.clear()
                stablePoints = null
            }
            return stablePoints
        }
        missStreak = 0

        val maxDrift = frameWidth * maxCornerDriftRatio
        if (recentSamples.isNotEmpty() && !isClose(recentSamples.last(), newPoints, maxDrift)) {
            // Jumped somewhere new - probably a different candidate, not the
            // same object drifting - start over rather than blending the two.
            recentSamples.clear()
        }

        recentSamples.add(newPoints)
        if (recentSamples.size > requiredSamples) {
            recentSamples.removeAt(0)
        }

        if (recentSamples.size >= requiredSamples) {
            stablePoints = averageCorners(recentSamples)
        }
        return stablePoints
    }

    fun reset() {
        recentSamples.clear()
        missStreak = 0
        stablePoints = null
    }

    private fun isClose(a: Array<Point>, b: Array<Point>, maxDistance: Double): Boolean {
        for (i in a.indices) {
            val dx = a[i].x - b[i].x
            val dy = a[i].y - b[i].y
            if (sqrt(dx * dx + dy * dy) > maxDistance) return false
        }
        return true
    }

    private fun averageCorners(samples: List<Array<Point>>): Array<Point> {
        return Array(4) { i ->
            val avgX = samples.sumOf { it[i].x } / samples.size
            val avgY = samples.sumOf { it[i].y } / samples.size
            Point(avgX, avgY)
        }
    }
}
