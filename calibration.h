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
	//���ر궨ͼƬ
	void LoadCalibImage(const vector<string>& imgPathList);
	void setPatternSize(cv::Size size);
	void cornerPointExtraction();
	void cameraCalibration();
	void calculateError();
	//���������������ԭͼ��
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
	// ͼ���С
	cv::Size imageSize_;
	//�궨����ÿ�С�ÿ�еĽǵ���������ͼƬ�еı궨�����ڽǵ���Ϊ8*6
	cv::Size patternSize_ = cv::Size(11, 8);
	//����ͼƬ�Ľǵ���Ϣ
	vector<vector<cv::Point2f>> cornerPointsOfAllImages_; 
	//����궨ʱ�õ���������
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	//��������ͼƬ�Ľǵ����ά����
	vector<vector<cv::Point3f>> objectPoints_; 
	// ����ξ���H������Ӧ�Ծ���
	cv::Mat cameraMatrix_;
	//�������5������ϵ����k1,k2,p1,p2,k3
	cv::Mat distCoefficients_;
	//ÿ��ͼ���ƽ��������t��������ͼƬ�ģ�����һ��ͼ���
	std::vector<cv::Mat> tvecsMat;
	//ÿ��ͼ�����ת�������޵������ת������������ͼƬ�ģ�����һ��ͼ���
	std::vector<cv::Mat> rvecsMat;
	//���б궨ͼ������ܺ�
	double totalErr_ = 0.0;
	//ÿ��ͼ���ƽ�����
	double* Err_All{};
	//ƴ�ӻ�׼����תλ������
	cv::Mat Base_RVecsMat;
	cv::Mat Base_TVecsMat;
};