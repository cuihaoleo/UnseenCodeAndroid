#include "us_barcode.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <utility>
#include <vector>

static cv::Mat hadamard(int n) {
    assert(n > 0 && (n & (n - 1)) == 0);

    cv::Mat h(1, 1, CV_32F);
    h.at<float>(0, 0) = 1.0;

    while ((n >>= 1) > 0) {
        cv::Mat dup1[2] = {h, h};
        cv::Mat dup2[2] = {h, -h};
        cv::Mat rows[2];
        cv::vconcat(dup1, 2, rows[0]);
        cv::vconcat(dup2, 2, rows[1]);
        cv::hconcat(rows, 2, h);
    }

    return h;
}

cv::Mat get_base(int width) {
    const cv::Point2f pts1[] = {
            cv::Point2f(-0.5, -0.5),
            cv::Point2f(7.5, -0.5),
            cv::Point2f(-0.5, 7.5),
    };
    const cv::Point2f pts2[] = {
            cv::Point2f(0.0, width/2.0),
            cv::Point2f(width/2.0, 0.0),
            cv::Point2f(width/2.0, width),
    };
    return cv::getAffineTransform(pts1, pts2);
}

std::pair<cv::Mat, cv::Mat> preproc(const cv::Mat &bgr) {
    cv::Mat bgr_f32, resized;

    if (bgr.type() == CV_8UC3) {
        bgr.convertTo(bgr_f32, CV_32FC3, 1.0 / 255.0);
    } else if (bgr.type() == CV_32FC3) {
        bgr_f32 = bgr;
    } else {
        throw "Not implemented!";
    }

    cv::resize(bgr_f32, resized, cv::Size(DEST_WIDTH, DEST_WIDTH));
    cv::GaussianBlur(resized, resized, cv::Size(9, 9), 0, 0);

    float TRANS[3][3] = {{0.072169, 0.212671, 0.715160},
                         {0.950227, 0.019334, 0.119193},
                         {0.180423, 0.412453, 0.357580}};
    cv::Mat trans_mat = cv::Mat(3, 3, CV_32FC1, TRANS).t();
    cv::Mat seq = resized.reshape(1, DEST_WIDTH * DEST_WIDTH) * trans_mat;
    cv::Mat xyz = seq.reshape(3, DEST_WIDTH);  //cv::cvtColor(blur, xyz, cv::COLOR_BGR2XYZ);
    std::vector<cv::Mat> channels(3);
    cv::split(xyz, channels);

    //cv::Mat yu = channels[0];
    cv::Mat ca = channels[1];
    cv::Mat cb = channels[2];

    return std::make_pair(ca, cb);
};

std::vector<bool> decode(const cv::Mat &ca,
                         const cv::Mat &cb,
                         int n_block) {
    assert(ca.size == cb.size);
    assert(ca.rows == ca.cols);
    auto dest_width = ca.rows;
    auto patch_width = dest_width / n_block;
    auto mat = get_base(patch_width);
    std::vector<bool> msg;
    auto hada = hadamard(8);
    cv::Mat ca_nm, cb_nm;

    std::vector<int> step1, step2;
    for (int i=0; i<n_block; i++)
        step1.push_back(dest_width * i / n_block);
    for (int i=1; i<n_block*2-1; i+=2)
        step2.push_back(dest_width * i / (2 * n_block));

    //cv::normalize(ca - cv::mean(ca), ca_nm, 1.0, 0.0, cv::NORM_L2);
    //cv::normalize(cb - cv::mean(cb), cb_nm, 1.0, 0.0, cv::NORM_L2);
    for (auto step: {step1, step2}) {
        for (auto sy: step) {
            for (auto sx: step) {
                auto roi = cv::Rect(sx, sy, patch_width, patch_width);
                auto pca = ca(roi).clone();
                auto pcb = cb(roi).clone();

                //pca -= cv::mean(pca);
                //pcb -= cv::mean(pcb);
                cv::normalize(pca - cv::mean(pca), pca, 1.0, 0.0, cv::NORM_L2);
                cv::normalize(pcb - cv::mean(pcb), pcb, 1.0, 0.0, cv::NORM_L2);

                cv::Mat b8ca, b8cb;
                cv::warpAffine(pca, b8ca, mat, cv::Size(8, 8), cv::WARP_INVERSE_MAP);
                cv::warpAffine(pcb, b8cb, mat, cv::Size(8, 8), cv::WARP_INVERSE_MAP);

                auto diff = b8ca - b8cb;
                cv::Mat dht = cv::abs(hada * diff * hada);

                bool bit = (dht.at<float>(2, 0) + dht.at<float>(3, 0) >
                            dht.at<float>(0, 2) + dht.at<float>(0, 3));
                msg.push_back(bit);
            }
        }
    }

    msg.pop_back();
    return msg;
}

std::vector<cv::Point> detect_box(const cv::Mat &img,
                                  double min_area_ratio,
                                  double rect_similarity,
                                  int approx_poly_round) {
    cv::Mat gray;
    cv::Mat binary;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    //cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(gray, binary, 180, 255, cv::THRESH_BINARY);
    cv::findContours(binary, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double min_area = img.cols * img.rows * min_area_ratio;

    std::vector<cv::Point> best_ct;
    cv::RotatedRect best_box(cv::Point(0, 0), cv::Size(0, 0), 0);

    for (auto ct: contours) {
        std::vector<cv::Point> hull;
        cv::convexHull(ct, hull);
        double area = cv::contourArea(hull);

        // approx
        cv::RotatedRect b = cv::minAreaRect(hull);

        if (b.size.area() > best_box.size.area()
            && area / b.size.area() > rect_similarity
            && area > min_area) {
            best_box = b;
            best_ct = hull;
        }
    }

    if (best_ct.size() == 0)
        return best_ct;

    // exact
    std::vector<cv::Point> approx;
    double min_epsilon = 0.0;
    double max_epsilon = 0.2;
    double ref_len = std::sqrt(best_box.size.area());

    for (int round=0; round<approx_poly_round; round++) {
        double epsilon = (min_epsilon + max_epsilon) / 2.0;
        cv::approxPolyDP(best_ct, approx, epsilon * ref_len, true);
        // printf("epsilon: %f, num of approx: %lu\n", epsilon, approx.size());

        if (approx.size() == 4)
            break;
        else if (approx.size() > 4)
            min_epsilon = epsilon;
        else
            max_epsilon = epsilon;

        approx.empty();
    }

    if (approx.size() != 4)
        return std::vector<cv::Point>();

    int first_pt = 0;
    int min = INT_MAX;
    for (int i=0; i<4; i++) {
        int sum = approx[i].x + approx[i].y;
        if (sum < min) {
            min = sum;
            first_pt = i;
        }
    }

    std::rotate(approx.begin(), approx.begin() + first_pt, approx.end());
    return approx;
}