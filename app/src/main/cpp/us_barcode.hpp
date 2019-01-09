#ifndef _US_BARCODE_HPP_INCLUDED_
#define _US_BARCODE_HPP_INCLUDED_

extern "C" {
#include "unseencode.h"
}

#include "opencv2/core.hpp"
#include <utility>
#include <vector>

const int DEST_WIDTH = 1200;
const int BLOCK_N = 17;
const int CLEN = (BLOCK_N * BLOCK_N) + (BLOCK_N - 1) * (BLOCK_N - 1);
const int PREFIX_LEN = (CLEN - TOTAL_BIT) / 2;
const int POSTFIX_LEN = (CLEN - TOTAL_BIT) - PREFIX_LEN;

std::vector<cv::Point> detect_box(const cv::Mat &img,
                                  double min_area_ratio=0.1,
                                  double rect_similarity=0.85,
                                  int approx_poly_round=10);
std::vector<bool> decode(const cv::Mat &ca, const cv::Mat &cb, int n_block);
std::pair<cv::Mat, cv::Mat> preproc(const cv::Mat &bgr);

#endif