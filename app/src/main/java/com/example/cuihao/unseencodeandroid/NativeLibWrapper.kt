package com.example.cuihao.unseencodeandroid

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.lang.RuntimeException
import java.util.*

const val DEST_WIDTH = 1200
const val BLOCK_N = 17
const val MLEN = 511
const val CLEN = (BLOCK_N * BLOCK_N) + (BLOCK_N - 1) * (BLOCK_N - 1)
const val PREFIX_LEN = (CLEN - MLEN) / 2
const val POSTFIX_LEN = (CLEN - MLEN) - PREFIX_LEN

const val BORDER_WIDTH = 14
const val TARGET_SIZE = 512

class NativeLibWrapper {
    external fun stringFromJNI(): String?
    external fun decodeJNI(addr: Long): ByteArray
    external fun detectBoxJNI(addr: Long): DoubleArray?
    external fun unseenCodeDecodeJNI(arr: ByteArray): String?

    fun detectBox(bgr: Mat): Array<Point>? {
        val buf = detectBoxJNI(bgr.nativeObjAddr) ?: return null
        return Array(4) { Point(buf[it*2], buf[it*2+1]) }
    }

    fun correctImage(bgr: Mat, pts: Array<Point>): Mat {
        if (pts.size != 4) {
            throw RuntimeException("pts.size != 4")
        }

        val transformed = Mat(TARGET_SIZE + BORDER_WIDTH*2, TARGET_SIZE + BORDER_WIDTH*2, CvType.CV_8UC3)
        val src_mat = MatOfPoint2f()
        src_mat.fromList(pts.asList())
        val dst_mat = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(transformed.width() - 1.0, 0.0),
            Point(transformed.width() - 1.0, transformed.height() - 1.0),
            Point(0.0, transformed.height() - 1.0)
        )

        val matrix = Imgproc.getPerspectiveTransform(src_mat, dst_mat)
        Imgproc.warpPerspective(bgr, transformed, matrix, transformed.size())

        return Mat(transformed, Rect(BORDER_WIDTH, BORDER_WIDTH, TARGET_SIZE, TARGET_SIZE))
    }

    fun decodeUnseenCode(bgr: Mat): BitSet {
        val msg_bits = decodeJNI(bgr.nativeObjAddr)
        val ret = BitSet(msg_bits.size)
        msg_bits.indices.forEach{ ret[it] = (msg_bits[it] != 0.toByte()) }
        return ret
    }

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }
}