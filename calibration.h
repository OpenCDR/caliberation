#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

class CalibrateCamera {
public:
	CalibrateCamera();
	~CalibrateCamera();
	//加载标定图片
	void LoadCalibImage(const vector<string>& imgPathList);
	void setPatternSize(cv::Size size);
	void cornerPointExtraction();
	void cameraCalibration();
	void calculateError();
	//畸变矫正（不保存原图像）
	void adjustImage(const cv::Mat& image) const;
	void run();
	//Get
	cv::Mat GetCameraMatrix();
	cv::Mat GetDistCoeff();
	double GetAverage_err();
	int GetImageNum();
	double* GetErr_ALL();
	cv::Mat GetBaseRvecMat();
	cv::Mat GetBaseTvecMat();
private:
	//
	vector<cv::Mat> imgList_;
	// 图像大小
	cv::Size imageSize_;
	//标定板上每行、每列的角点数；测试图片中的标定板上内角点数为8*6
	cv::Size patternSize_ = cv::Size(11, 8);
	//所有图片的角点信息
	vector<vector<cv::Point2f>> cornerPointsOfAllImages_; 
	//保存标定时用的像素坐标
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	//保存所有图片的角点的三维坐标
	vector<vector<cv::Point3f>> objectPoints_; 
	// 内外参矩阵，H——单应性矩阵
	cv::Mat cameraMatrix_;
	//摄像机的5个畸变系数：k1,k2,p1,p2,k3
	cv::Mat distCoefficients_;
	//每幅图像的平移向量，t，若干张图片的，不是一张图像的
	std::vector<cv::Mat> tvecsMat;
	//每幅图像的旋转向量（罗德里格旋转向量，若干张图片的，不是一张图像的
	std::vector<cv::Mat> rvecsMat;
	//所有标定图像误差总和
	double totalErr_ = 0.0;
	//每幅图像的平均误差
	double* Err_All{};
	//拼接基准的旋转位移向量
	cv::Mat Base_RVecsMat;
	cv::Mat Base_TVecsMat;
};