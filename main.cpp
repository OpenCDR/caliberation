#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "calibration.h"

using namespace std;

int main() {
	ifstream inImgPath("calibdata.txt"); //�궨����ͼ���ļ���·��
	vector<string> imgPathList;
	vector<string>::iterator p;
	string temp;
	if (!inImgPath.is_open())
		cout << "û���ҵ��ļ�" << endl;
	//��ȡ�ļ��б����ͼƬ�ļ�·�����������������
	while (getline(inImgPath, temp))
		imgPathList.push_back(temp);
	CalibrateCamera camera;
	camera.LoadCalibImage(imgPathList);
	camera.run();
	return 0;
}