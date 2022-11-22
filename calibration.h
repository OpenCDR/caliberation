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
	//���ر궨ͼƬ
	void loadCalibImage(const vector<string>& imgPathList);
	void setPatternSize(cv::Size size);
	void baseCalculate(cv::Mat rgbImage);
	cv::Point3f getWorldPoints(const cv::Point2f& inPoints, const cv::Mat& rvecs, const cv::Mat& tvecs) const;
	cv::Point3f myRotate(const cv::Point3f& p, cv::Vec6f line, int k) const;
	cv::Point3f getCameraPoints(const cv::Point2f& inPoints) const;
	//���������������ԭͼ��
	void adjustImage(const cv::Mat& image) const;
	void run();
	//Get
	cv::Mat getCameraMatrix();
	cv::Mat getDistCoeff();
	double getAverageErr() const;
	cv::Mat getBaseRvecsMat();
	cv::Mat getBaseTvecsMat();
	//��������ͼƬ�Ľǵ����ά����
	vector<vector<cv::Point3f>> objectPoints_;
private:
	//
	vector<cv::Mat> imgList_;
	// ͼ���С
	cv::Size imageSize_;
	//�궨����ÿ�С�ÿ�еĽǵ���������ͼƬ�еı궨�����ڽǵ���Ϊ8*6
	cv::Size patternSize_ = cv::Size(10, 8);
	//����ͼƬ�Ľǵ���Ϣ
	vector<vector<cv::Point2f>> cornerPointsOfAllImages_;
	//����궨ʱ�õ���������
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	// ����ξ���H������Ӧ�Ծ���
	cv::Mat cameraMatrix_;
	//�������5������ϵ����k1,k2,p1,p2,k3
	cv::Mat distCoefficients_;
	//ÿ��ͼ���ƽ��������t��������ͼƬ�ģ�����һ��ͼ���
	std::vector<cv::Mat> tvecsMat_;
	//ÿ��ͼ�����ת�������޵������ת������������ͼƬ�ģ�����һ��ͼ���
	std::vector<cv::Mat> rvecsMat_;
	//���б궨ͼ������ܺ�
	double totalErr_ = 0.0;
	//ÿ��ͼ���ƽ�����
	double* errAll_{};
	//ƴ�ӻ�׼����תλ������
	cv::Mat baseRVecsMat_;
	cv::Mat baseTVecsMat_;

	void cornerPointExtraction();
	void cameraCalibration();
	void calculateError();
};

class LaserPlane {
public:
	//��ƽ�����
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
	//�궨����ÿ�С�ÿ�еĽǵ���������ͼƬ�еı궨�����ڽǵ���Ϊ8*6
	cv::Size patternSize_ = cv::Size(10, 8);
	//�����ȡ�ı궨���ص�
	std::vector<std::vector<cv::Point2f>> imagePoint_;
	//����������ת������λ������������֮������׼����ϵת��
	vector<cv::Mat> rvecsMat_;
	vector<cv::Mat> tvecsMat_;
	//��������ת������׼�ĵ�����
	std::vector<cv::Point3f> points3ds_;

	double A = 0;
	double B = 0;
	double C = 0;
	double D = 0;
};
