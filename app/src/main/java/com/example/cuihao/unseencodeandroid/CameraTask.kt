package com.example.cuihao.unseencodeandroid

import android.graphics.Bitmap
import android.os.AsyncTask
import android.util.Log
import io.fotoapparat.Fotoapparat
import kotlinx.android.synthetic.main.activity_main.*
import org.jetbrains.anko.doAsync
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.lang.RuntimeException
import java.util.*

interface CameraTaskResponse {
    fun onNewResult(result: BitSet)
}

class CameraTask(val listener: CameraTaskResponse): AsyncTask<Fotoapparat, BitSet, Unit>() {
    private val TAG = CameraTask::class.qualifiedName

    override fun doInBackground(vararg fotoapparat: Fotoapparat?) {
        while (!isCancelled) {
            val photoResult = fotoapparat[0]?.takePicture()
            val bitmapPhoto = photoResult?.toBitmap()?.await()
            bitmapPhoto ?: continue

            doAsync {
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

                val pts = NativeLibWrapper().detectBox(bgr) ?: return@doAsync
                val roi = NativeLibWrapper().correctImage(bgr, pts)
                //saveImage("corr_$index.png", roi)
                //sample_text.text = "Photo $index processed!"
                val msg = NativeLibWrapper().decodeUnseenCode(roi)
                publishProgress(msg)
            }
        }
    }

    override fun onProgressUpdate(vararg values: BitSet?) {
        values.forEach {
            if (it != null) {
                listener.onNewResult(it)
            }
        }
    }
}