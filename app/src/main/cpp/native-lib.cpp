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
Java_com_example_cuihao_unseencodeandroid_MainActivity_decodeJNI(
        JNIEnv *env,
        jobject /* this */,
        long addrInMat) {
    cv::Mat *img = (cv::Mat*)addrInMat;
    cv::Mat bgr_f32, resized, blur, xyz;
    std::vector<cv::Mat> channels(3);

    (*img).convertTo(bgr_f32, CV_32FC3, 1.0/255.0);

    cv::resize(bgr_f32, resized, cv::Size(DEST_WIDTH, DEST_WIDTH));
    cv::GaussianBlur(resized, blur, cv::Size(9, 9), 0, 0);

    float TRANS[3][3] = {{0.072169, 0.212671, 0.715160}, {0.950227, 0.019334, 0.119193}, {0.180423, 0.412453, 0.357580}};
    cv::Mat trans_mat = cv::Mat(3, 3, CV_32FC1, TRANS).t();
    cv::Mat seq = blur.reshape(1, blur.cols*blur.rows) * trans_mat;
    xyz = seq.reshape(3, blur.rows);
    //cv::cvtColor(blur, xyz, cv::COLOR_BGR2XYZ);
    cv::split(xyz, channels);

    cv::Mat yu = channels[0];  // Y
    cv::Mat ca = channels[1];  // X
    cv::Mat cb = channels[2];  // Z

    std::vector<bool> msg = decode(ca, cb, BLOCK_N);
    jbyteArray result = env->NewByteArray((jsize)msg.size());
    jbyte buf[msg.size()];
    for (int i=0; i<msg.size(); i++)
        buf[i] = msg[i];

    env->SetByteArrayRegion(result, 0, (jsize)msg.size(), buf);
    return result;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_example_cuihao_unseencodeandroid_MainActivity_detectBoxJNI(
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
Java_com_example_cuihao_unseencodeandroid_MainActivity_unseenCodeDecodeJNI(
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
Java_com_example_cuihao_unseencodeandroid_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
