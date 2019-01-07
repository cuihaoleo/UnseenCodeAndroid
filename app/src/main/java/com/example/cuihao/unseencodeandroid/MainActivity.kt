package com.example.cuihao.unseencodeandroid

import android.app.Activity
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.facedetector.Rectangle
import io.fotoapparat.log.logcat
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.preview.Frame
import io.fotoapparat.result.BitmapPhoto
import io.fotoapparat.selector.*
import kotlinx.android.synthetic.main.activity_main.*
import org.jetbrains.anko.doAsync
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import java.lang.RuntimeException
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import org.opencv.core.*
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Mat
import java.util.*

const val DEST_WIDTH = 1200
const val BLOCK_N = 17
const val MLEN = 511
const val CLEN = (BLOCK_N * BLOCK_N) + (BLOCK_N - 1) * (BLOCK_N - 1)
const val PREFIX_LEN = (CLEN - MLEN) / 2
const val POSTFIX_LEN = (CLEN - MLEN) - PREFIX_LEN

class MainActivity : Activity(), View.OnClickListener {

    private val TAG = MainActivity::class.qualifiedName

    val configuration by lazy {
        CameraConfiguration(
            frameProcessor = this::processPreview,
            jpegQuality = highestQuality(),
            sensorSensitivity = highestSensorSensitivity()
        )
    }

    val fotoapparat by lazy {
        Fotoapparat(
            context = this,
            view = camera_view,                   // view which will draw the camera preview
            scaleType = ScaleType.CenterCrop,    // (optional) we want the preview to fill the view
            lensPosition = back(),               // (optional) we want back camera
            cameraConfiguration = configuration, // (optional) define an advanced configuration
            logger = logcat()
            // cameraErrorCallback = { error -> }   // (optional) log fatal errors
        )
    }

    private fun saveImage(name: String, mat: Mat) {
        val path = File("${getExternalFilesDir(null)}")
        path.mkdirs()
        val file = File(path, name)
        val bool = Imgcodecs.imwrite(file.toString(), mat)

        if (bool) {
            Log.i(TAG, "SUCCESS writing image to external storage")
        } else {
            Log.i(TAG, "Fail writing image to external storage")
        }
    }

    private fun processPreview(frame: Frame) {
        val yuv = Mat(
            frame.size.height + frame.size.height / 2,
            frame.size.width,
            CvType.CV_8UC1
        )
        yuv.put(0, 0, frame.image)
        val bgr = Mat(frame.size.height, frame.size.width, CvType.CV_8UC3)
        Imgproc.cvtColor(yuv, bgr, Imgproc.COLOR_YUV2BGR_NV21)
        Log.d(TAG,"  FRAME: ${frame.size.height} x ${frame.size.width}, ${frame.rotation}")

        when (frame.rotation) {
            90 -> Core.rotate(bgr, bgr, Core.ROTATE_90_COUNTERCLOCKWISE)
            180 -> Core.rotate(bgr, bgr, Core.ROTATE_180)
            270 -> Core.rotate(bgr, bgr, Core.ROTATE_90_CLOCKWISE)
        }
        /*saveImage("image.png", bgr)
        Log.d(TAG,"PREVIEW: ${bgr.height()} x ${bgr.width()}")

        val pts = detectBox(bgr) ?: return
        val roi = correctImage(bgr, pts)
        val msg = decodeUnseenCode(roi)
        saveImage("corr.png", roi)
        Log.d(TAG, "MSG: $msg")*/

        val pts = detectBox(bgr) ?: return
        this.runOnUiThread{
            rectangles_view.setRectangles(List(4) {
                Rectangle(
                    (pts[it].x / bgr.cols()).toFloat() - 0.01F,
                    (pts[it].y / bgr.rows()).toFloat() - 0.01F,
                    0.02F, 0.02F
                )
            })
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OPENCV BAD")
        }

        sample_text.text = stringFromJNI()
        button.setOnClickListener(this)
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()
    }

    override fun onStop() {
        super.onStop()
        fotoapparat.stop()
    }

    fun collectPhoto() {
        doAsync {
            val dup = 9
            val photos = Array(dup) {
                Log.d(TAG,"Photo $it go...")
                val photoResult = fotoapparat.takePicture()
                val bitmapPhoto = photoResult.toBitmap().await()
                Log.d(TAG,"Photo $it captured!")
                sample_text.text = "Photo $it captured!"
                bitmapPhoto
            }

            val vote = Array(CLEN){ 0 }
            fotoapparat.stop()
            photos.forEachIndexed{ index, bitmapPhoto -> Int
                val rgba = Mat()
                val bgr = Mat()
                val bmp32 = bitmapPhoto?.bitmap?.copy(Bitmap.Config.ARGB_8888, true)
                bmp32 ?: throw RuntimeException("BAD BITMAP")
                Utils.bitmapToMat(bmp32, rgba)
                Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)

                when (bitmapPhoto.rotationDegrees) {
                    90 -> Core.rotate(bgr, bgr, Core.ROTATE_90_COUNTERCLOCKWISE)
                    180 -> Core.rotate(bgr, bgr, Core.ROTATE_180)
                    270 -> Core.rotate(bgr, bgr, Core.ROTATE_90_CLOCKWISE)
                }

                val pts = detectBox(bgr) ?: return@forEachIndexed
                val roi = correctImage(bgr, pts)
                saveImage("corr_$index.png", roi)
                sample_text.text = "Photo $index processed!"
                val msg = decodeUnseenCode(roi)
                vote.indices.forEach{
                    vote[it] += if (msg[it]) 1 else 0
                }
            }

            val finalMessage = vote.map{ if (it * 2 > dup) 1.toByte() else 0.toByte() }
            finalMessage.subList(PREFIX_LEN, PREFIX_LEN + MLEN)
            val msg = unseenCodeDecodeJNI(finalMessage.toByteArray())
            sample_text.text = msg
            fotoapparat.start()
        }
    }

    override fun onClick(v: View?) {
        collectPhoto()
        /*val photoResult = fotoapparat.takePicture()
        photoResult
            .toBitmap()
            .whenAvailable { bitmapPhoto: BitmapPhoto? ->
                val rgba = Mat()
                val bgr = Mat()
                val bmp32 = bitmapPhoto?.bitmap?.copy(Bitmap.Config.ARGB_8888, true)
                bmp32 ?: throw RuntimeException("BAD BITMAP")
                Utils.bitmapToMat(bmp32, rgba)
                Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)

                when (bitmapPhoto.rotationDegrees) {
                    90 -> Core.rotate(bgr, bgr, Core.ROTATE_90_COUNTERCLOCKWISE)
                    180 -> Core.rotate(bgr, bgr, Core.ROTATE_180)
                    270 -> Core.rotate(bgr, bgr, Core.ROTATE_90_CLOCKWISE)
                }

                //saveImage("big_image.png", bgr)
                val pts = detectBox(bgr) ?: return@whenAvailable
                val roi = correctImage(bgr, pts)
                val msg = decodeUnseenCode(roi)
                //saveImage("big_corr_$count.png", roi)
                Log.d(TAG, "BIG_MSG: $msg")
            }*/
    }

    private external fun stringFromJNI(): String?
    private external fun decodeJNI(addr: Long): ByteArray
    private external fun detectBoxJNI(addr: Long): DoubleArray?
    private external fun unseenCodeDecodeJNI(arr: ByteArray): String?

    private fun detectBox(bgr: Mat): Array<Point>? {
        val buf = detectBoxJNI(bgr.nativeObjAddr) ?: return null
        return Array(4) { Point(buf[it*2], buf[it*2+1]) }
    }

    private fun correctImage(bgr: Mat, pts: Array<Point>): Mat {
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

    private fun decodeUnseenCode(bgr: Mat): BitSet {
        val msg_bits = decodeJNI(bgr.nativeObjAddr)
        val ret = BitSet(msg_bits.size)
        msg_bits.indices.forEach{ ret[it] = (msg_bits[it] != 0.toByte()) }
        return ret
    }

    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
        }

        private const val BORDER_WIDTH = 14
        private const val TARGET_SIZE = 512
    }
}