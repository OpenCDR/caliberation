#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#define PI 3.1415926

using namespace std;


class CalibrateCamera {
public:
	void calRealPoint(std::vector<std::vector<cv::Point3f>>& obj, int imageNum, float squareSize, float z = 0) const;
	CalibrateCamera();
	~CalibrateCamera();
	//加载标定图片
	void loadCalibImage(const vector<string>& imgPathList);
	void setPatternSize(cv::Size size);
	void baseCalculate(cv::Mat rgbImage);
	cv::Point3f getWorldPoints(const cv::Point2f& inPoints, const cv::Mat& rvecs, const cv::Mat& tvecs) const;
	cv::Point3f myRotate(const cv::Point3f& p, cv::Vec6f line, int k) const;
	cv::Point3f getCameraPoints(const cv::Point2f& inPoints) const;
	//畸变矫正（不保存原图像）
	void adjustImage(const cv::Mat& image) const;
	void run();
	//Get
	cv::Mat getCameraMatrix();
	cv::Mat getDistCoeff();
	double getAverageErr() const;
	cv::Mat getBaseRvecsMat();
	cv::Mat getBaseTvecsMat();
	//保存所有图片的角点的三维坐标
	vector<vector<cv::Point3f>> objectPoints_;
private:
	//
	vector<cv::Mat> imgList_;
	// 图像大小
	cv::Size imageSize_;
	//标定板上每行、每列的角点数；测试图片中的标定板上内角点数为8*6
	cv::Size patternSize_ = cv::Size(10, 8);
	//所有图片的角点信息
	vector<vector<cv::Point2f>> cornerPointsOfAllImages_;
	//保存标定时用的像素坐标
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	// 内外参矩阵，H——单应性矩阵
	cv::Mat cameraMatrix_;
	//摄像机的5个畸变系数：k1,k2,p1,p2,k3
	cv::Mat distCoefficients_;
	//每幅图像的平移向量，t，若干张图片的，不是一张图像的
	std::vector<cv::Mat> tvecsMat_;
	//每幅图像的旋转向量（罗德里格旋转向量，若干张图片的，不是一张图像的
	std::vector<cv::Mat> rvecsMat_;
	//所有标定图像误差总和
	double totalErr_ = 0.0;
	//每幅图像的平均误差
	double* errAll_{};
	//拼接基准的旋转位移向量
	cv::Mat baseRVecsMat_;
	cv::Mat baseTVecsMat_;

	void cornerPointExtraction();
	void cameraCalibration();
	void calculateError();
};

class LaserPlane {
public:
	//光平面参数
	void calculateLaserPlane();
	void loadBoard(const vector<cv::Mat>& boardNoLaser, CalibrateCamera& camera);
	void loadLaser(vector<cv::Mat>& boardLaser, CalibrateCamera& camera);
	//get
	vector<cv::Mat> getRVecsMat();
	vector<cv::Mat> getTVecsMat();
	double getA() const;
	double getB() const;
	double getC() const;
	double getD() const;
private:
	//标定板上每行、每列的角点数；测试图片中的标定板上内角点数为8*6
	cv::Size patternSize_ = cv::Size(10, 8);
	//保存读取的标定像素点
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	//保存计算的旋转向量和位移向量，用于之后的向基准坐标系转换
	vector<cv::Mat> rvecsMat_;
	vector<cv::Mat> tvecsMat_;
	//保存所有转化到基准的点坐标
	std::vector<cv::Point3f> points3ds_;

	double A = 0;
	double B = 0;
	double C = 0;
	double D = 0;
};
