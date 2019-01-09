#include <jni.h>
#include <string>

extern "C" {
#include "err_chk.h"
#include "unseencode.h"
}

#include "us_barcode.hpp"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_cuihao_unseencodeandroid_NativeLibWrapper_decodeJNI(
        JNIEnv *env,
        jobject /* this */,
        long addrInMat) {
    cv::Mat img = *(cv::Mat*)addrInMat;
    auto p = preproc(img);
    std::vector<bool> msg = decode(p.first, p.second, BLOCK_N);

    jbyteArray result = env->NewByteArray((jsize)msg.size());
    jbyte buf[msg.size()];
    for (int i=0; i<msg.size(); i++)
        buf[i] = msg[i];

    env->SetByteArrayRegion(result, 0, (jsize)msg.size(), buf);
    return result;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_example_cuihao_unseencodeandroid_NativeLibWrapper_detectBoxJNI(
        JNIEnv *env,
        jobject /* this */,
        long addrInMat) {
    cv::Mat *img = (cv::Mat*)addrInMat;
    std::vector<cv::Point> pts = detect_box(*img);

    if (pts.size() == 0)
        return nullptr;

    jdoubleArray result = env->NewDoubleArray(8);
    jdouble buf[8];
    for (int i=0; i<8; i++)
        buf[i] = (i % 2) ? pts[i/2].y : pts[i/2].x;

    env->SetDoubleArrayRegion(result, 0, 8, buf);
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_cuihao_unseencodeandroid_NativeLibWrapper_unseenCodeDecodeJNI(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray input) {
    jbyte* arr = env->GetByteArrayElements(input, 0);
    char *str_c = unseencode_decode((uint8_t*)arr);
    env->ReleaseByteArrayElements(input, arr, 0);

    if (str_c == nullptr)
        return nullptr;
    else {
        jstring result = env->NewStringUTF(str_c);
        free(str_c);
        return result;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_cuihao_unseencodeandroid_NativeLibWrapper_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
