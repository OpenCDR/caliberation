#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <vector>


class ProcessTool {
public:
	std::vector<cv::Point2d> averageLine(cv::Mat img, const cv::Point2d& leftUp, const cv::Point2f& rightDown) const;
	std::vector<cv::Point2d> stegerLine(cv::Mat img0, int col = 3, int row = 9, int sqrtX = 1, int sqrtY = 1,
	                                    int threshold = 80, float distance = 0.5, bool isFloat = false) const;
private:
	void showLine(const std::vector<cv::Point2d>& points, cv::Mat image) const;
};

#endif
