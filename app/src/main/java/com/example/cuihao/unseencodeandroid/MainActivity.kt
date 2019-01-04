package com.example.cuihao.unseencodeandroid

import android.app.Activity
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.facedetector.Rectangle
import io.fotoapparat.log.logcat
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.preview.Frame
import io.fotoapparat.result.BitmapPhoto
import io.fotoapparat.selector.back
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import java.lang.RuntimeException
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import org.opencv.core.*
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Mat
import kotlin.text.StringBuilder


class MainActivity : Activity(), View.OnClickListener {

    private val TAG = MainActivity::class.qualifiedName

    val configuration by lazy {
        CameraConfiguration(
            frameProcessor = this::processPreview
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
        saveImage("image.png", bgr)

        Log.d(TAG,"PREVIEW: ${bgr.height()} x ${bgr.width()}")
        val buf = detectBoxJNI(bgr.nativeObjAddr) ?: return

        val pts = Array(4) { Point(buf[it*2], buf[it*2+1]) }

        val rects = Array(4) {
            Rectangle(
                (pts[it].x / bgr.cols()).toFloat() - 0.01F,
                (pts[it].y / bgr.rows()).toFloat() - 0.01F,
                0.02F, 0.02F
            )
        }

        this.runOnUiThread{ rectangles_view.setRectangles(rects.asList()) }

        val BORDER_WIDTH = 14
        val TARGET_SIZE = 512

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

        val roi = Mat(transformed, Rect(BORDER_WIDTH, BORDER_WIDTH, TARGET_SIZE, TARGET_SIZE))
        saveImage("corr.png", roi)

        val msg_bits = decodeJNI(roi.nativeObjAddr) ?: return;
        val msg_str = StringBuilder()
        msg_bits.forEach {
            msg_str.append(it);
        }

        Log.d(TAG, "MSG: $msg_str")
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

    override fun onClick(v: View?) {
        val photoResult = fotoapparat.takePicture()
        photoResult
            .toBitmap()
            .whenAvailable { bitmapPhoto: BitmapPhoto? ->
                val mat = Mat()
                val bmp32 = bitmapPhoto?.bitmap?.copy(Bitmap.Config.ARGB_8888, true)
                bmp32 ?: throw RuntimeException("BAD BITMAP")
                Utils.bitmapToMat(bmp32, mat);
                Log.d(TAG, "${mat.height()} x ${mat.width()}")
            }
    }

    private external fun stringFromJNI(): String
    private external fun decodeJNI(addr: Long): ByteArray?
    private external fun detectBoxJNI(addr: Long): DoubleArray?

    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
        }
    }
}