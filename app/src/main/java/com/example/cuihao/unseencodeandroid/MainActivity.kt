package com.example.cuihao.unseencodeandroid

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.facedetector.Rectangle
import io.fotoapparat.log.logcat
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.preview.Frame
import io.fotoapparat.result.BitmapPhoto
import io.fotoapparat.selector.*
import kotlinx.android.synthetic.main.activity_main.*
import org.jetbrains.anko.custom.async
import org.jetbrains.anko.design.snackbar
import org.jetbrains.anko.doAsync
import org.jetbrains.anko.progressDialog
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import java.lang.RuntimeException
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import org.opencv.core.*
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Mat
import java.lang.annotation.Native
import java.util.*
import java.util.concurrent.Executors

class MainActivity : Activity(), View.OnClickListener, CameraTaskResponse {

    private val TAG = MainActivity::class.qualifiedName

    val configuration by lazy {
        CameraConfiguration(
            frameProcessor = this::processPreview,
            jpegQuality = highestQuality(),
            sensorSensitivity = highestSensorSensitivity(),
            pictureResolution = firstAvailable(
                {Resolution(2448, 2448)},
                highestResolution())
        )
    }

    val fotoapparat by lazy {
        Fotoapparat(
            context = this,
            view = camera_view,                     // view which will draw the camera preview
            scaleType = ScaleType.CenterCrop,       // (optional) we want the preview to fill the view
            lensPosition = back(),                  // (optional) we want back camera
            cameraConfiguration = configuration,    // (optional) define an advanced configuration
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

        val pts = NativeLibWrapper().detectBox(bgr) ?: return
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

        sample_text.text = NativeLibWrapper().stringFromJNI()
        button.setOnClickListener(this)
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()

        fotoapparat.getCapabilities().whenAvailable {
            it?.pictureResolutions?.forEach{
                Log.d(TAG, "SUPPORT: $it")
            }
        }
    }

    override fun onStop() {
        super.onStop()
        fotoapparat.stop()
    }

    var mTask: CameraTask? = null
    private fun collectPhoto() {
        mTask = CameraTask(this)
        mTask?.execute(fotoapparat)
        button.isEnabled = false
    }

    val mVote = Array(CLEN){ 0 }
    var mResultCount = 0

    override fun onNewResult(result: BitSet) {
        mVote.indices.forEach{
            mVote[it] += if (result[it]) 1 else 0
        }

        if ((mResultCount++) and 1 == 1) {
            val finalMessage = mVote.map{ if (it * 2 > mResultCount) 1.toByte() else 0.toByte() }
            val validCode = finalMessage.subList(PREFIX_LEN, PREFIX_LEN + MLEN)
            val msg = NativeLibWrapper().unseenCodeDecodeJNI(validCode.toByteArray())

            if (msg != null) {
                //sample_text.text = msg
                mTask?.cancel(false)

                mResultCount = 0
                mVote.fill(0)

                button.isEnabled = true

                AlertDialog.Builder(this@MainActivity)
                    .setTitle("UnseenCode")
                    .setMessage(msg)
                    .create()
                    .show()
            }
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


    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
        }
    }
}