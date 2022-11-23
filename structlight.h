// use for measurement
#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

class Structlight
{
public:
	Structlight();
	~Structlight();
	cv::Mat cameraIntrinsic;//������ڲξ���
	cv::Mat distCoeffs;//5������ϵ��
	double lightPlaneFormular[4];//��ƽ����� Ax + By + Cz = D

	cv::Mat Rw;//��ת����
	cv::Mat Tw;//λ������

	void readParameters();
};

