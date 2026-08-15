/**
    Adapted from Document-Scanning-Android-SDK (zynkware). Reset to a
    byte-faithful copy of the original detection algorithm (see
    md/status-2026-08-14.md for why) - the only intentional omission is
    getScannedBitmap/getContourEdgePoints/getPoint, which depend on the
    original's Bitmap<->Mat conversion and PerspectiveTransformation classes
    (crop/perspective-transform infrastructure, not used by this live-only
    detection screen - ScanSurfaceView.kt in the source calls
    detectLargestQuadrilateral directly, exactly as this screen does).
    Original file:
    https://github.com/zynkware/Document-Scanning-Android-SDK/blob/master/DocumentScanner/src/main/java/com/zynksoftware/documentscanner/common/utils/OpenCvNativeBridge.kt

    Copyright 2020 ZynkSoftware SRL

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package com.example.insolescanplan2.docscan

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.*
import kotlin.math.*

internal class Quadrilateral(val contour: MatOfPoint2f, val points: Array<Point>)

// Every field independently toggleable so each guard's effect can be
// isolated while testing, rather than guessing which one helps or hurts.
// Applied in this order (matches the order they're actually evaluated in
// findQuadrilateral/isReasonableQuadrilateral):
//  1. edgeMarginGuardEnabled - reject if any corner is too close to the
//     frame's own boundary (the camera view edge itself is a stable false
//     "edge" that live-frame detection can otherwise lock onto).
//  2. oppositeSideGuardEnabled - opposite sides (top/bottom, left/right)
//     must be reasonably similar in length.
//  3. minAreaGuardEnabled - detected quad must be a large-enough fraction
//     of the whole frame to plausibly be the real object, not a small
//     background-texture blob.
//  4. aspectRatioGuardEnabled - overall long-side/short-side can't be too
//     extreme (rejects near-degenerate slivers from a broken contour).
//  5. centerPriorityEnabled - among surviving candidates, prefer whichever
//     one's centroid sits closest to the frame's own center. Not part of
//     the original algorithm at all - with this off, behavior matches the
//     original exactly: the first valid candidate in area order wins.
// All default to false - the baseline is universal "any 4-edge shape"
// scanning, matching the original source's own unmodified algorithm.
internal data class GuardConfig(
    val edgeMarginGuardEnabled: Boolean = false,
    val oppositeSideGuardEnabled: Boolean = false,
    val minAreaGuardEnabled: Boolean = false,
    val aspectRatioGuardEnabled: Boolean = false,
    val centerPriorityEnabled: Boolean = false
)

internal class OpenCvNativeBridge {

    companion object {
        private const val ANGLES_NUMBER = 4
        private const val EPSILON_CONSTANT = 0.02
        private const val CLOSE_KERNEL_SIZE = 10.0
        private const val CANNY_THRESHOLD_LOW = 75.0
        private const val CANNY_THRESHOLD_HIGH = 200.0
        private const val CUTOFF_THRESHOLD = 155.0
        private const val TRUNCATE_THRESHOLD = 150.0
        private const val NORMALIZATION_MIN_VALUE = 0.0
        private const val NORMALIZATION_MAX_VALUE = 255.0
        private const val BLURRING_KERNEL_SIZE = 5.0
        private const val FIRST_MAX_CONTOURS = 10

        // Shape-sanity tolerances, both scale-free (ratios, not absolute
        // pixels) so they hold regardless of frame resolution or how close
        // the object is.
        //
        // Opposite sides (top vs bottom, left vs right): the shorter one
        // must be at least this fraction of the longer one. 0.5 allows a
        // fairly steep viewing angle before rejecting.
        private const val OPPOSITE_SIDE_MIN_RATIO = 0.5
        // Overall long-side/short-side of the whole quadrilateral - rejects
        // near-degenerate slivers a broken contour tends to produce, while
        // still allowing genuinely elongated rectangular objects.
        private const val MAX_ASPECT_RATIO = 3.0
        // Detected quad area, as a fraction of the whole frame's area -
        // below this it's more likely a small background-texture blob than
        // the actual object.
        private const val MIN_AREA_RATIO = 0.06
        // A candidate with any corner within this fraction of the frame's
        // width/height from the frame's own boundary is rejected outright.
        private const val EDGE_MARGIN_RATIO = 0.03
    }

    // patch from Udayraj123 (https://github.com/Udayraj123/LiveEdgeDetection)
    fun detectLargestQuadrilateral(src: Mat, guardConfig: GuardConfig = GuardConfig()): Quadrilateral? {
        // Captured before any in-place processing below - blur/threshold/etc.
        // mutate pixel values, never the Mat's dimensions.
        val frameWidth = src.cols().toDouble()
        val frameHeight = src.rows().toDouble()
        val frameCenterX = frameWidth / 2.0
        val frameCenterY = frameHeight / 2.0
        val frameArea = frameWidth * frameHeight
        val destination = Mat()
        Imgproc.blur(src, src, Size(BLURRING_KERNEL_SIZE, BLURRING_KERNEL_SIZE))

        Core.normalize(src, src, NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE, Core.NORM_MINMAX)

        Imgproc.threshold(src, src, TRUNCATE_THRESHOLD, NORMALIZATION_MAX_VALUE, Imgproc.THRESH_TRUNC)
        Core.normalize(src, src, NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE, Core.NORM_MINMAX)

        Imgproc.Canny(src, destination, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW)

        Imgproc.threshold(destination, destination, CUTOFF_THRESHOLD, NORMALIZATION_MAX_VALUE, Imgproc.THRESH_TOZERO)

        Imgproc.morphologyEx(
            destination, destination, Imgproc.MORPH_CLOSE,
            Mat(Size(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), CvType.CV_8UC1, Scalar(NORMALIZATION_MAX_VALUE)),
            Point(-1.0, -1.0), 1
        )

        val largestContour: List<MatOfPoint>? = findLargestContours(destination)
        if (null != largestContour) {
            return findQuadrilateral(largestContour, frameCenterX, frameCenterY, frameArea, frameWidth, frameHeight, guardConfig)
        }
        return null
    }

    private fun findQuadrilateral(
        mContourList: List<MatOfPoint>,
        frameCenterX: Double,
        frameCenterY: Double,
        frameArea: Double,
        frameWidth: Double,
        frameHeight: Double,
        guardConfig: GuardConfig
    ): Quadrilateral? {
        var best: Quadrilateral? = null
        var bestDistanceToCenter = Double.MAX_VALUE

        for (c in mContourList) {
            val c2f = MatOfPoint2f(*c.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, EPSILON_CONSTANT * peri, true)
            val points = approx.toArray()

            // select biggest 4 angles polygon
            val candidate: Pair<MatOfPoint2f, Array<Point>>? = if (approx.rows() == ANGLES_NUMBER) {
                approx to sortPoints(points)
            } else if (approx.rows() == 5) {
                // if document has a bent corner
                var shortestDistance = Int.MAX_VALUE.toDouble()
                var shortestPoint1: Point? = null
                var shortestPoint2: Point? = null

                var diagonal = 0.toDouble()
                var diagonalPoint1: Point? = null
                var diagonalPoint2: Point? = null

                for (i in 0 until 4) {
                    for (j in i + 1 until 5) {
                        val d = distance(points[i], points[j])
                        if (d < shortestDistance) {
                            shortestDistance = d
                            shortestPoint1 = points[i]
                            shortestPoint2 = points[j]
                        }
                        if (d > diagonal) {
                            diagonal = d
                            diagonalPoint1 = points[i]
                            diagonalPoint2 = points[j]
                        }
                    }
                }

                val trianglePointWithHypotenuse: Point? = points.toList().minus(arrayListOf(shortestPoint1, shortestPoint2, diagonalPoint1, diagonalPoint2))[0]

                val newPoint = if (trianglePointWithHypotenuse!!.x > shortestPoint1!!.x && trianglePointWithHypotenuse.x > shortestPoint2!!.x &&
                    trianglePointWithHypotenuse.y > shortestPoint1.y && trianglePointWithHypotenuse.y > shortestPoint2.y
                ) {
                    Point(min(shortestPoint1.x, shortestPoint2.x), min(shortestPoint1.y, shortestPoint2.y))
                } else if (trianglePointWithHypotenuse.x < shortestPoint1.x && trianglePointWithHypotenuse.x < shortestPoint2!!.x &&
                    trianglePointWithHypotenuse.y > shortestPoint1.y && trianglePointWithHypotenuse.y > shortestPoint2.y
                ) {
                    Point(max(shortestPoint1.x, shortestPoint2.x), min(shortestPoint1.y, shortestPoint2.y))
                } else if (trianglePointWithHypotenuse.x < shortestPoint1.x && trianglePointWithHypotenuse.x < shortestPoint2!!.x &&
                    trianglePointWithHypotenuse.y < shortestPoint1.y && trianglePointWithHypotenuse.y < shortestPoint2.y
                ) {
                    Point(max(shortestPoint1.x, shortestPoint2.x), max(shortestPoint1.y, shortestPoint2.y))
                } else if (trianglePointWithHypotenuse.x > shortestPoint1.x && trianglePointWithHypotenuse.x > shortestPoint2!!.x &&
                    trianglePointWithHypotenuse.y < shortestPoint1.y && trianglePointWithHypotenuse.y < shortestPoint2.y
                ) {
                    Point(min(shortestPoint1.x, shortestPoint2.x), max(shortestPoint1.y, shortestPoint2.y))
                } else {
                    Point(0.0, 0.0)
                }

                val sortedPoints = sortPoints(arrayOf(trianglePointWithHypotenuse, diagonalPoint1!!, diagonalPoint2!!, newPoint))
                val newApprox = MatOfPoint2f()
                newApprox.fromArray(*sortedPoints)
                newApprox to sortedPoints
            } else {
                null
            }

            if (candidate == null) continue
            val (candidateContour, sortedPoints) = candidate

            if (guardConfig.edgeMarginGuardEnabled && touchesFrameEdge(sortedPoints, frameWidth, frameHeight)) continue
            if (!isReasonableQuadrilateral(sortedPoints, frameArea, guardConfig)) continue

            if (!guardConfig.centerPriorityEnabled) {
                // Original algorithm's own behavior: first valid candidate
                // in area order wins outright, no center comparison at all.
                return Quadrilateral(candidateContour, sortedPoints)
            }

            val centroidX = sortedPoints.sumOf { it.x } / sortedPoints.size
            val centroidY = sortedPoints.sumOf { it.y } / sortedPoints.size
            val distanceToCenter = sqrt((centroidX - frameCenterX).pow(2.0) + (centroidY - frameCenterY).pow(2.0))
            if (distanceToCenter < bestDistanceToCenter) {
                bestDistanceToCenter = distanceToCenter
                best = Quadrilateral(candidateContour, sortedPoints)
            }
        }
        return best
    }

    private fun touchesFrameEdge(sortedPoints: Array<Point>, frameWidth: Double, frameHeight: Double): Boolean {
        val marginX = frameWidth * EDGE_MARGIN_RATIO
        val marginY = frameHeight * EDGE_MARGIN_RATIO
        return sortedPoints.any { p ->
            p.x <= marginX || p.x >= frameWidth - marginX ||
                p.y <= marginY || p.y >= frameHeight - marginY
        }
    }

    // sortedPoints follows sortPoints' convention: [0]=top-left, [1]=top-right,
    // [2]=bottom-right, [3]=bottom-left.
    private fun isReasonableQuadrilateral(sortedPoints: Array<Point>, frameArea: Double, guardConfig: GuardConfig): Boolean {
        val top = distance(sortedPoints[0], sortedPoints[1])
        val right = distance(sortedPoints[1], sortedPoints[2])
        val bottom = distance(sortedPoints[2], sortedPoints[3])
        val left = distance(sortedPoints[3], sortedPoints[0])
        // Always-on structural check, not a tunable "guard" - a zero-length
        // side means the 4 points are degenerate (at least two coincide).
        if (top <= 0.0 || right <= 0.0 || bottom <= 0.0 || left <= 0.0) return false

        if (guardConfig.oppositeSideGuardEnabled) {
            val topBottomRatio = min(top, bottom) / max(top, bottom)
            val leftRightRatio = min(left, right) / max(left, right)
            if (topBottomRatio < OPPOSITE_SIDE_MIN_RATIO || leftRightRatio < OPPOSITE_SIDE_MIN_RATIO) return false
        }

        if (guardConfig.minAreaGuardEnabled) {
            // Shoelace formula for a simple (non-self-intersecting)
            // quadrilateral - sortedPoints are already in consistent
            // TL->TR->BR->BL winding order.
            val quadArea = 0.5 * abs(
                sortedPoints[0].x * (sortedPoints[1].y - sortedPoints[3].y) +
                    sortedPoints[1].x * (sortedPoints[2].y - sortedPoints[0].y) +
                    sortedPoints[2].x * (sortedPoints[3].y - sortedPoints[1].y) +
                    sortedPoints[3].x * (sortedPoints[0].y - sortedPoints[2].y)
            )
            if (quadArea / frameArea < MIN_AREA_RATIO) return false
        }

        if (guardConfig.aspectRatioGuardEnabled) {
            val avgLongSide = max((top + bottom) / 2.0, (left + right) / 2.0)
            val avgShortSide = min((top + bottom) / 2.0, (left + right) / 2.0)
            if (avgLongSide / avgShortSide > MAX_ASPECT_RATIO) return false
        }

        return true
    }

    private fun distance(p1: Point, p2: Point): Double {
        return sqrt((p1.x - p2.x).pow(2.0) + (p1.y - p2.y).pow(2.0))
    }

    private fun sortPoints(src: Array<Point>): Array<Point> {
        val srcPoints: ArrayList<Point> = ArrayList(src.toList())
        val result = arrayOf<Point?>(null, null, null, null)
        val sumComparator: Comparator<Point> = Comparator<Point> { lhs, rhs -> (lhs.y + lhs.x).compareTo(rhs.y + rhs.x) }
        val diffComparator: Comparator<Point> = Comparator<Point> { lhs, rhs -> (lhs.y - lhs.x).compareTo(rhs.y - rhs.x) }

        // top-left corner = minimal sum
        result[0] = Collections.min(srcPoints, sumComparator)
        // bottom-right corner = maximal sum
        result[2] = Collections.max(srcPoints, sumComparator)
        // top-right corner = minimal difference
        result[1] = Collections.min(srcPoints, diffComparator)
        // bottom-left corner = maximal difference
        result[3] = Collections.max(srcPoints, diffComparator)
        return result.map {
            it!!
        }.toTypedArray()
    }

    private fun findLargestContours(inputMat: Mat): List<MatOfPoint>? {
        val mHierarchy = Mat()
        val mContourList: List<MatOfPoint> = ArrayList()
        // finding contours - as we are sorting by area anyway, we can use RETR_LIST - faster than RETR_EXTERNAL.
        Imgproc.findContours(inputMat, mContourList, mHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        // Convert the contours to their Convex Hulls i.e. removes minor nuances in the contour
        val mHullList: MutableList<MatOfPoint> = ArrayList()
        val tempHullIndices = MatOfInt()
        for (i in mContourList.indices) {
            Imgproc.convexHull(mContourList[i], tempHullIndices)
            mHullList.add(hull2Points(tempHullIndices, mContourList[i]))
        }
        // Release mContourList as its job is done
        for (c in mContourList) {
            c.release()
        }
        tempHullIndices.release()
        mHierarchy.release()
        if (mHullList.size != 0) {
            mHullList.sortWith { lhs, rhs ->
                Imgproc.contourArea(rhs).compareTo(Imgproc.contourArea(lhs))
            }
            return mHullList.subList(0, min(mHullList.size, FIRST_MAX_CONTOURS))
        }
        return null
    }

    private fun hull2Points(hull: MatOfInt, contour: MatOfPoint): MatOfPoint {
        val indexes = hull.toList()
        val points: MutableList<Point> = ArrayList()
        val ctrList = contour.toList()
        for (index in indexes) {
            points.add(ctrList[index])
        }
        val point = MatOfPoint()
        point.fromList(points)
        return point
    }

    fun contourArea(approx: MatOfPoint2f): Double {
        return Imgproc.contourArea(approx)
    }
}
