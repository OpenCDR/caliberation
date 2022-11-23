#include "structure_light.h"
#include "struct_light_calib.h"
#include "structlight.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <opencv2/core/types.hpp>

using namespace std;

int MAX_ITER = 100000;
double eps = 0.0000001;

//定义一些全局变量作为标定的靶标参数
int imageCount = 11; //图像的数量
cv::Size patternSize(7, 7); //靶标（圆的数量）
cv::Size2f patternLength(37.5, 37.5); //两个圆形标记之间的距离
//patternType:
//0:circle;
//1:chessboard;
bool isCircle = true;

int main() {
	structure_light lineStructureLight(imageCount, patternSize, patternLength);

	double reProjectionError = 0.0;
	cameraCalib(lineStructureLight, reProjectionError); //摄像机标定
	cout << "Camera calibration reProjection error: " << reProjectionError << " pixels." << endl;

	//光条中心提取
	//steger光条中心提取算法
	stegerLine(lineStructureLight);

	//结构光光条直线拟合与相交点坐标提取
	vector<vector<cv::Point2f>> intersectPoint; //光平面和靶标平面上的点
	crossPoint(lineStructureLight, intersectPoint);
	//交比不变求取光平面三维坐标
	crossRatio(lineStructureLight, intersectPoint);

	//拟合光平面
	lightPlainFitting(lineStructureLight);

	// 输出点云的数量
	cout << lineStructureLight.lightPlanePoint.size() << endl;

	// 将点云存储在一个TXT文件中
	ofstream outfile;
	outfile.open("pointCloud.txt", ios::binary | ios::app | ios::in | ios::out);
	for (auto& k : lineStructureLight.lightPlanePoint) {
		outfile << k.x << " ";
		outfile << k.y << " ";
		outfile << k.z << "\n";
	}
	outfile.close(); //关闭文件，保存文件
	//输出标定结果
	lineStructureLight.outputResult();

	//对标定结果进行评估（方法：类反投影）
	Structlight structLightMeasure;
	structLightMeasure.readParameters();
	vector<vector<float>> backProjectError;
	estimateError2(lineStructureLight, structLightMeasure, backProjectError);
	cout << "finish!" << endl;
	return 0;

}
