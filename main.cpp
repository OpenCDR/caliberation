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
	ifstream inImgPath("calibdata.txt"); //标定所用图像文件的路径
	vector<string> imgPathList;
	vector<string>::iterator p;
	string temp;
	if (!inImgPath.is_open())
		cout << "没有找到文件" << endl;
	//读取文件中保存的图片文件路径，并存放在数组中
	while (getline(inImgPath, temp))
		imgPathList.push_back(temp);
	CalibrateCamera camera;
	camera.LoadCalibImage(imgPathList);
	camera.run();
	return 0;
}