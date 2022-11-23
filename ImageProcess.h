#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


class ProcessTool {
public:
	std::vector<cv::Point2d> averageLine(cv::Mat img, const cv::Point2d& leftUp, const cv::Point2f& rightDown) const;
	std::vector<cv::Point2d> stegerLine(cv::Mat img0, int col = 15, int row = 15, int sqrtX = 4, int sqrtY = 4,
	                                    int threshold = 80, float distance = 0.5, bool isFloat = false) const;
	static bool comp(const cv::Point& a, const cv::Point& b);
	static bool neighborAnalysis(const cv::Point2i& point, cv::Mat img, const int threshold);
private:
	void showLine(const std::vector<cv::Point2d>& points, cv::Mat image) const;
};

#endif
