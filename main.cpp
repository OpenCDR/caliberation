#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "calibration.h"
#include "ImageProcess.h"

using namespace std;

int main() {
	ifstream inImgPath("calibdata.txt"); //标定所用图像文件的路径
	vector<string> imgPathList;
	string temp;
	if (!inImgPath.is_open())
		cout << "没有找到文件" << endl;
	//读取文件中保存的图片文件路径，并存放在数组中
	while (getline(inImgPath, temp))
		imgPathList.push_back(temp);
	auto image = cv::imread("LaserImages2/base.bmp");
	CalibrateCamera camera;
	LaserPlane laserPlane;
	camera.loadCalibImage(imgPathList);
	camera.run();
	camera.baseCalculate(image);
	ifstream noLaserImgPath("Plane_Board_NoLaser.txt"); //标定所用图像文件的路径
	vector<cv::Mat> noLaserImageList;
	while (getline(noLaserImgPath, temp)) {
		noLaserImageList.push_back(cv::imread(temp));
	}
	ifstream laserImgPath("Plane_Board_Laser.txt"); //标定所用图像文件的路径
	vector<cv::Mat> laserImageList;
	while (getline(laserImgPath, temp)) {
		laserImageList.push_back(cv::imread(temp));
	}
	laserPlane.loadBoard(noLaserImageList, camera);
	laserPlane.loadLaser(laserImageList, camera);
	laserPlane.calculateLaserPlane();
	image = cv::imread("LaserImages2/Image_20221121194108563.bmp");
	cv::Mat r = camera.getBaseRvecsMat();
	cv::Mat t = camera.getBaseTvecsMat();

	ProcessTool tool;
	std::vector<std::vector<float>> pointss;
	auto points = tool.averageLine(image, cv::Point2d(0, 0), cv::Point2d(image.cols, image.rows));
	for (auto& point : points) {
		cv::Point3f points3d = camera.getWorldPoints(cv::Point2f(point.x, point.y), r, t);
		points3d.z = (laserPlane.getD() - laserPlane.getA() * points3d.x - laserPlane.getB() * points3d
			.y) / laserPlane.getC();
		std::vector<float> kk = { points3d.x, points3d.y, points3d.z };
		std::cout << points3d.x << " " << points3d.y << " " << points3d.z << endl;
		pointss.push_back(kk);
	}
	return 0;
}