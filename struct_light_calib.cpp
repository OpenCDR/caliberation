#include "struct_light_calib.h"
#include "structure_light.h"
#include "structlight.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <cmath>

extern int MAX_ITER;
extern double eps;
extern int imageCount;
extern cv::Size patternSize;
extern cv::Size2f patternLength;
//extern cv::Size imageSize;
//patternType:
//0:circle;
//1:chessboard;
extern bool isCircle;

//摄像机标定
int cameraCalib(structure_light& a, double& reProjectionError) {
	string format = ".jpg";
	cv::Size imageSize;
	for (int i = 0; i < a.imageCount; i++) {
		string index = to_string(i);
		string name = "./calib_picture/calib" + index + format;
		cv::Mat pic = cv::imread(name);

		cv::Mat greyImage;
		cvtColor(pic, greyImage, cv::COLOR_BGR2GRAY);
		imageSize.width = greyImage.cols;
		imageSize.height = greyImage.rows;

		bool result = true;
		vector<cv::Point2f> targetPoint;
		//圆形靶标提取圆心点
		//提取靶标上圆斑的圆心
		if (isCircle) {
			if (0 == findCirclesGrid(greyImage, a.patternSize, targetPoint)) {
				result = false;
				a.isRecognize.push_back(result);
				cout << "false-calib" << endl;
				continue;
			}
			result = true;
			a.isRecognize.push_back(result);
			a.calibImagePoint.push_back(targetPoint);
			cout << "true-calib" << endl;
		}
		//棋盘格靶标提取角点
		else {
			if (0 == findChessboardCorners(greyImage, a.patternSize, targetPoint)) {
				result = false;
				a.isRecognize.push_back(result);
				continue;
			}
			result = true;
			a.isRecognize.push_back(result);
			find4QuadCornerSubpix(greyImage, targetPoint, cv::Size(5, 5));
			a.calibImagePoint.push_back(targetPoint);
			drawChessboardCorners(greyImage, patternSize, targetPoint, true);
			imwrite("./save/Corner.jpg", greyImage);
		}
	}

	a.generateCalibBoardPoint();
	/* 摄像头标定得到相机内参和畸变矩阵以及外参（旋转矩阵R和平移矩阵T）*/
	double rms = calibrateCamera(a.calibBoardPoint, a.calibImagePoint, imageSize, a.cameraMatrix, a.distCoeffs, a.R,
	                             a.T);
	cout << "Re-projection error:" << rms << endl;

	a.Rw = a.R[0];
	a.Tw = a.T[0];
	/* 重投影评估单目摄像头标定精度 */
	double err = 0.0;
	double meanErr = 0.0;
	double totalErr = 0.0;
	int i;
	vector<cv::Point2f> reProjectionPoint;

	for (i = 0; i < a.calibBoardPoint.size(); i++) {
		vector<cv::Point3f> tempPointSet = a.calibBoardPoint[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, a.R[i], a.T[i], a.cameraMatrix, a.distCoeffs, reProjectionPoint);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<cv::Point2f> tempImagePoint = a.calibImagePoint[i];
		auto tempImagePointMat = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
		auto imagePoints2Mat = cv::Mat(1, reProjectionPoint.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++) {
			imagePoints2Mat.at<cv::Vec2f>(0, j) = cv::Vec2f(reProjectionPoint[j].x, reProjectionPoint[j].y);
			tempImagePointMat.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(imagePoints2Mat, tempImagePointMat, cv::NORM_L2);

		meanErr = err / (a.patternSize.width * a.patternSize.height);
		totalErr += meanErr;
	}
	reProjectionError = totalErr / a.calibBoardPoint.size();
	return 0;
}

//################# RANSAC算法拟合直线相关函数 #########################//
//生成[0,1]之间符合均匀分布的数
double uniformRandom() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }

//直线样本中两随机点位置不能太近
bool verifyComposition(const vector<cv::Point2f>& pts) {
	const cv::Point2d pt1 = pts[0];
	const cv::Point2d pt2 = pts[1];
	if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
		return false;
	return true;
}

//根据点集拟合直线ax+by+c=0，res为残差
void calcLinePara(const vector<cv::Point2f>& pts, double& a, double& b, double& c, double& res) {
	res = 0;
	cv::Vec4f line;
	vector<cv::Point2f> ptsF;
	for (const auto& pt : pts)
		ptsF.push_back(pt);

	fitLine(ptsF, line, cv::DIST_L2, 0, 1e-2, 1e-2);

	a = line[1];
	b = -line[0];
	c = line[0] * line[3] - line[1] * line[2];

	for (auto& pt : pts) {
		const double resid = fabs(pt.x * a + pt.y * b + c);
		res += resid;
	}
	res /= pts.size();
}

//得到直线拟合样本，即在直线采样点集上随机选2个点
bool getSample(const vector<int> set, vector<int>& sset) {
	if (set.size() > 2) {
		int i[2];
		do
			for (int& n : i)
				n = static_cast<int>(uniformRandom() * (set.size() - 1));
		while (!(i[1] != i[0]));
		for (int& n : i)
			sset.push_back(n);
	}
	else
		return false;
	return true;
}


//RANSAC直线拟合
void fitLineRANSAC(vector<cv::Point2f> ptSet, double& a, double& b, double& c, vector<bool>& inlierFlag) {

	bool stopLoop = false;
	int maximum = 0; //最大内点数

	//最终内点标识及其残差
	inlierFlag = vector<bool>(ptSet.size(), false);
	vector<double> resids_(ptSet.size(), 3);
	int sampleCount = 0;
	int N = 500;

	double res = 0;

	// RANSAC
	srand(static_cast<unsigned>(time(nullptr))); //设置随机数种子
	vector<int> ptsID;
	for (unsigned int i = 0; i < ptSet.size(); i++)
		ptsID.push_back(i);
	while (N > sampleCount && !stopLoop) {
		vector<bool> inlierstemp;
		vector<double> residualStemp;
		vector<int> ptss;
		int inlierCount = 0;
		if (!getSample(ptsID, ptss)) {
			stopLoop = true;
			continue;
		}

		vector<cv::Point2f> pt_sam;
		pt_sam.push_back(ptSet[ptss[0]]);
		pt_sam.push_back(ptSet[ptss[1]]);

		if (!verifyComposition(pt_sam)) {
			++sampleCount;
			continue;
		}

		// 计算直线方程
		calcLinePara(pt_sam, a, b, c, res);
		//内点检验
		for (unsigned int i = 0; i < ptSet.size(); i++) {
			constexpr double residualError = 2.99;
			const cv::Point2f pt = ptSet[i];
			double resid = fabs(pt.x * a + pt.y * b + c);
			residualStemp.push_back(resid);
			inlierstemp.push_back(false);
			if (resid < residualError) {
				++inlierCount;
				inlierstemp[i] = true;
			}
		}
		// 找到最佳拟合直线
		if (inlierCount >= maximum) {
			maximum = inlierCount;
			resids_ = residualStemp;
			inlierFlag = inlierstemp;
		}
		// 更新RANSAC迭代次数，以及内点概率
		if (inlierCount == 0)
			N = 500;
		else {
			const double epsilon = 1.0 - static_cast<double>(inlierCount) / static_cast<double>(ptSet.size()); //野值点比例
			constexpr double p = 0.99; //所有样本中存在1个好样本的概率
			constexpr double s = 2.0;
			N = static_cast<int>(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
		}
		++sampleCount;
	}

	//利用所有内点重新拟合直线
	vector<cv::Point2f> pset;
	for (unsigned int i = 0; i < ptSet.size(); i++)
		if (inlierFlag[i])
			pset.push_back(ptSet[i]);

	calcLinePara(pset, a, b, c, res);
}

//##################### RANSAC算法拟合直线部分完成 ######################


//提取结构光光条中心点（steger算法）
void stegerLine(structure_light& a) {
	for (int k = 0; k < a.imageCount; k++) {
		if (a.isRecognize[k] == false)
			continue;

		cv::Mat lightLineImage;
		extratLightLine(k, lightLineImage, a);

		imwrite("./save/image.jpg", lightLineImage);
		//一阶偏导数
		cv::Mat m1, m2;
		m1 = (cv::Mat_<float>(1, 2) << 1, -1);
		m2 = (cv::Mat_<float>(2, 1) << 1, -1);

		cv::Mat dx, dy;
		filter2D(lightLineImage, dx, CV_32FC1, m1);
		filter2D(lightLineImage, dy, CV_32FC1, m2);

		//二阶偏导数
		cv::Mat m3, m4, m5;
		m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1); //二阶x偏导
		m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1); //二阶y偏导
		m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1); //二阶xy偏导

		cv::Mat dxx, dyy, dxy;
		filter2D(lightLineImage, dxx, CV_32FC1, m3);
		filter2D(lightLineImage, dyy, CV_32FC1, m4);
		filter2D(lightLineImage, dxy, CV_32FC1, m5);

		vector<cv::Point2f> oriImagePoint;
		vector<cv::Point2f> subPixelImagePoint;
		//hessian矩阵提取光条中心
		for (int i = 0; i < lightLineImage.cols; i++)
			for (int j = 0; j < lightLineImage.rows; j++)
				//通过灰度值确定光条像素

				if (lightLineImage.at<uchar>(j, i) != 0) {
					cv::Mat hessian(2, 2, CV_32FC1);
					hessian.at<float>(0, 0) = dxx.at<float>(j, i);
					hessian.at<float>(0, 1) = dxy.at<float>(j, i);
					hessian.at<float>(1, 0) = dxy.at<float>(j, i);
					hessian.at<float>(1, 1) = dyy.at<float>(j, i);

					cv::Mat eValue, eVectors;
					eigen(hessian, eValue, eVectors);

					double nx, ny;
					double fmaxD = 0;
					if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0))) {
						nx = eVectors.at<float>(0, 0);
						ny = eVectors.at<float>(0, 1);
						fmaxD = eValue.at<float>(0, 0);
					}
					else {
						nx = eVectors.at<float>(1, 0);
						ny = eVectors.at<float>(1, 1);
						fmaxD = eValue.at<float>(1, 0);
					}

					double t = -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) / (nx * nx * dxx.at<float>(j, i) + 2
						* nx * ny * dyy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));

					if ((fabs(t * nx) <= 0.5) && (fabs(t * ny) <= 0.5)) {
						cv::Point2i oriPoint;
						oriPoint.x = i;
						oriPoint.y = j;
						oriImagePoint.push_back(oriPoint);

						cv::Point2f subPixelPoint;
						subPixelPoint.x = i + t * nx;
						subPixelPoint.y = j + t * ny;
						subPixelImagePoint.push_back(subPixelPoint); //亚像素的光条中心点
					}
				}
		a.lightPlaneOriImagePoint.push_back(oriImagePoint);
		//a.lightPlaneSubPixelImagePoint.push_back(subPixelImagePoint);
		/* RANSAC算法拟合直线，滤除坏点 */
		//Mat ImageRANSAC = imread(".\\light_picture\\light1.jpg");
		double A, B, C;
		vector<bool> inliers;
		fitLineRANSAC(subPixelImagePoint, A, B, C, inliers);
		//for (unsigned int i = 0; i < subPixelImagePoint.size(); i++) {
		//	if (inliers[i])
		//		circle(ImageRANSAC, subPixelImagePoint[i], 3, Scalar(0, 255, 0), 3, 16);
		//	else
		//		circle(ImageRANSAC, subPixelImagePoint[i], 3, Scalar(0, 0, 255), 3, 16);
		//}

		B = B / A;
		C = C / A;
		A = A / A;

		//绘制直线
		//Point2d ptStart, ptEnd;
		//ptStart.x = 0;
		//ptStart.y = -(A*ptStart.x + C) / B;
		//ptEnd.x = -(B*ptEnd.y + C) / A;
		//ptEnd.y = 0;
		//line(ImageRANSAC, ptStart, ptEnd, Scalar(0, 0, 0), 1, 16);
		//cout << "A:" << A << " " << "B:" << B << " " << "C:" << C << " " << endl;
		//for (int k = 0; k < subPixelImagePoint.size(); k++)//在图中画出些点，然后保存图片
		//{
		//	circle(ImageRANSAC, subPixelImagePoint[k], 1, Scalar(0, 255, 0));
		//}
		//imwrite(".\\save\\imageRANSAC.jpg", ImageRANSAC);
		//计算所有点到拟合直线的距离
		//Mat ImageNO = imread(".\\light_picture\\light1.jpg");
		vector<cv::Point2f> subPixelImagePointTemp;
		for (auto& i : subPixelImagePoint) {
			float distance = abs(A * i.x + B * i.y + C) / sqrt(A * A + B * B);
			//cout << "点" << i << "到直线的距离 :  "<<  distance  << endl;
			if (distance > 1);
			else
				//circle(ImageNO, subPixelImagePoint[i], 1, Scalar(0, 255, 0));
				subPixelImagePointTemp.push_back(i);
		}
		subPixelImagePoint = subPixelImagePointTemp;
		a.lightPlaneSubPixelImagePoint.push_back(subPixelImagePoint);
		//imwrite(".\\save\\ImageNO.jpg", ImageNO);
		//Mat imagetemp = imread(".\\light_picture\\light1.jpg");
		//for (int ii = 0; ii < subPixelImagePoint.size(); ii++)
		//{
		//	circle(imagetemp, subPixelImagePoint[ii], 1, Scalar(0, 255, 0));
		//}
		//imwrite(".\\save\\imagetemp.jpg", imagetemp);
		//system("pause");

		/* RANSAC算法拟合直线部分结束 */

		cv::Mat lightLineImageResult = cv::Mat::zeros(lightLineImage.rows, lightLineImage.cols, CV_8U);
		for (auto& k : subPixelImagePoint) //在图中画出些点，然后保存图片
			circle(lightLineImageResult, k, 1, cv::Scalar(255, 255, 255));
		//float distance1 = abs(A*subPixelImagePoint[k].x + B * subPixelImagePoint[k].y + C) / sqrt(A*A + B * B);
		//cout << "点" << k << "到直线的距离 :  " << distance1 << endl;
		string format = ".jpg";
		string index = to_string(k);
		string name = "./save/result" + index + format;
		imwrite(name, lightLineImageResult);
		oriImagePoint.clear();
		subPixelImagePoint.clear();
	}
}

//结构光光条直线拟合与相交点坐标提取
//输入结构体类和点vector 结束之后vector就是光平面和靶标平面相交的直线上的点（7*7 = 49个 ）
void crossPoint(const structure_light& a, vector<vector<cv::Point2f>>& crossPoint) {
	for (int i = 0; i < imageCount; i++) {
		if (a.isRecognize[i] == false)
			continue;
		cv::Vec4f lightLine;
		vector<cv::Point2f> lightLinePoint = a.lightPlaneSubPixelImagePoint[i]; //取出第i张标定图像 的亚像素光条中心点
		fitLine(lightLinePoint, lightLine, cv::DIST_L2, 0, 1e-2, 1e-2); //将第i张图的 光条中心点进行直线拟合
		vector<cv::Point2f> cornerPoint = a.calibImagePoint[i]; //靶标平面上圆形标记中心的点（49个）的二维坐标
		vector<cv::Point2f> cornerLinePoint; //存储某一列的圆形标记的中心
		cv::Vec4f cornerLine; //一列圆标记中心拟合出的直线的参数
		vector<cv::Point2f> lightCornerPoint; //提取出的光条上的点
		//提取光条上的点，思路：取一列（或者一行）圆形标记的中心，拟合成直线，计算拟合出直线和光条的交点
		//遍历所有列7次
		for (int m = 0; m < patternSize.width; m++) {
			for (int n = 0; n < patternSize.height; n++)
				cornerLinePoint.push_back(cornerPoint[n * patternSize.width + m]); //取第m列圆形标记的中心
			fitLine(cornerLinePoint, cornerLine, cv::DIST_L2, 0, 1e-2, 1e-2); //将第m列的圆心拟合成直线
			//求出第m列圆心拟合出来的直线和光条直线的交点
			const double k1 = cornerLine[1] / cornerLine[0];
			const double b1 = cornerLine[3] - k1 * cornerLine[2];
			const double k2 = lightLine[1] / lightLine[0];
			const double b2 = lightLine[3] - k2 * lightLine[2];
			cv::Point2f temp;
			temp.x = (b2 - b1) / (k1 - k2);
			temp.y = k1 * temp.x + b1;
			//将交点压入向量
			lightCornerPoint.push_back(temp);
			cornerLinePoint.clear();
		}
		crossPoint.push_back(lightCornerPoint);
	}
}

//利用交比不变求取光平面三维坐标
void crossRatio(structure_light& light, vector<vector<cv::Point2f>>& crossPoint) {
	for (int i = 0; i < imageCount; i++) {
		if (light.isRecognize[i] == false)
			continue;
		vector<cv::Point2f> tempCrossPoint = crossPoint[i]; //某一张图中的光条上的7个点的二维坐标
		vector<cv::Point2f> tempCornerPoint = light.calibImagePoint[i]; //第i个靶标平面上圆形标记中心的点（49个）的二维坐标
		vector<cv::Point3f> tempWorldPoint = light.calibBoardPoint[i]; //第i个靶标平面上圆形标记中心的点（49个）的三维点坐标
		vector<cv::Point3f> tempCornerLinePoint; //用于存光条上的点的三维坐标
		//计算第i个靶标上的7个光条上的点的三维坐标（靶标坐标系下）循环7次
		for (int m = 0; m < tempCrossPoint.size(); m++) {
			//读取每条圆斑点形成直线上前三个特征点的坐标
			cv::Point2f a, b, c;
			cv::Point3f A, B, C;
			a = tempCornerPoint[m];
			b = tempCornerPoint[patternSize.width + m];
			c = tempCornerPoint[2 * patternSize.width + m];
			A = tempWorldPoint[m];
			B = tempWorldPoint[patternSize.width + m];
			C = tempWorldPoint[2 * patternSize.width + m];

			//计算交比
			double crossRatio = ((a.y - c.y) / (b.y - c.y)) / ((a.y - tempCrossPoint[m].y) / (b.y - tempCrossPoint[m].
				y));
			//double crossRatio = ((a.x - c.x) / (b.x - c.x)) / ((a.x - tempCrossPoint[m].x) / (b.x - tempCrossPoint[m].x));
			//已知4个点的图像坐标与3个点世界坐标，可以计算其中1点的世界坐标（现在已知三个圆形标记中心和一个光条上的点的图像坐标，和三个标记点的三维坐标）
			cv::Point3f crPoint;
			crPoint.x = m * patternLength.width;
			crPoint.y = (crossRatio * (B.y - C.y) * A.y - B.y * (A.y - C.y)) / (crossRatio * (B.y - C.y) - (A.y - C.y));
			//解出Y坐标
			crPoint.z = 0;
			//cout << "ccs Point:" << crPoint << endl;
			tempCornerLinePoint.push_back(crPoint);
		}

		cv::Mat rvec = light.R[i];
		cv::Mat T1 = light.T[i]; //靶标坐标系和摄像机坐标系之间的旋转平移矩阵
		cv::Mat rMat1, rMat2;
		Rodrigues(rvec, rMat1); //1x3 -> 3x3 旋转向量（1x3）与旋转矩阵（3x3），二者可以通过罗德里格斯相互转化

		//将靶标坐标系下的三维坐标转换成摄像机坐标系下的三维坐标
		vector<cv::Point3f> tempPoint;
		for (auto& n : tempCornerLinePoint) {
			cv::Point3f realPoint; //摄像机坐标系下的坐标，接下来利用旋转变换矩阵将坐标转化
			realPoint.x = rMat1.at<double>(0, 0) * n.x + rMat1.at<double>(0, 1) * n.y + T1.at<double>(0, 0);
			realPoint.y = rMat1.at<double>(1, 0) * n.x + rMat1.at<double>(1, 1) * n.y + T1.at<double>(1, 0);
			realPoint.z = rMat1.at<double>(2, 0) * n.x + rMat1.at<double>(2, 1) * n.y + T1.at<double>(2, 0);
			//cout << "wcs Point:" << realPoint << endl << endl;
			light.lightPlanePoint.push_back(realPoint);
		}
		tempCornerLinePoint.clear();
		tempWorldPoint.clear();
		tempCornerPoint.clear();
		tempCrossPoint.clear();
	}
}

//结构光的光平面拟合
void lightPlainFitting(structure_light& light) {
	cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1); //定义拟合所需要的三个矩阵
	cv::Mat B = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat X = cv::Mat::zeros(3, 1, CV_64FC1);

	A.at<double>(2, 2) = light.lightPlanePoint.size();
	for (const auto& i : light.lightPlanePoint) {
		A.at<double>(0, 0) += i.x * i.x;
		A.at<double>(0, 1) += i.x * i.y;
		A.at<double>(0, 2) += i.x;
		A.at<double>(1, 0) += i.x * i.y;
		A.at<double>(1, 1) += i.y * i.y;
		A.at<double>(1, 2) += i.y;
		A.at<double>(2, 0) += i.x;
		A.at<double>(2, 1) += i.y;
		B.at<double>(0, 0) += i.x * i.z;
		B.at<double>(1, 0) += i.y * i.z;
		B.at<double>(2, 0) += i.z;
	}

	solve(A, B, X);

	cout << "X:" << X << endl << endl;

	light.planeFormular[0] = -X.at<double>(0, 0); //x
	light.planeFormular[1] = -X.at<double>(1, 0); //y
	light.planeFormular[2] = 1; //z
	light.planeFormular[3] = X.at<double>(2, 0); //常数项
}

//差量法提取
void extratLightLine(int imageIndex, cv::Mat& outputImage, structure_light& a) {
	string num = to_string(imageIndex);
	string index = ".jpg";
	string name1 = "./calib_picture/calib" + num + index;
	string name2 = "./light_picture/light" + num + index;

	cv::Mat oriImage = cv::imread(name1);
	cv::Mat lightImage = cv::imread(name2);

	cvtColor(oriImage, oriImage, cv::COLOR_BGR2GRAY);
	cvtColor(lightImage, lightImage, cv::COLOR_BGR2GRAY);
	cv::Mat diff;
	absdiff(oriImage, lightImage, diff);
	imwrite("./save/diff.jpg", diff);

	cv::Mat element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	morphologyEx(diff, diff, cv::MORPH_OPEN, element1);
	imwrite("./save/open.jpg", diff);

	threshold(diff, diff, 30, 255, cv::THRESH_BINARY);
	imwrite("./save/threshold.jpg", diff);

	cv::Mat element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	erode(diff, diff, element2);
	imwrite("./save/erode.jpg", diff);

	vector<vector<cv::Point>> allMaskPoint;
	allMaskPoint.push_back(vector<cv::Point>());
	cv::Mat mask = cv::Mat::zeros(oriImage.size(), CV_8UC1);
	cv::Point2f tempPoint;
	cv::Point maskPoint;
	vector<cv::Point2f> tempPointSet = a.calibImagePoint[imageIndex];
	vector<cv::Point2f> maskPointSet;

	tempPoint = tempPointSet[0];
	maskPoint.x = cvRound(tempPoint.x);
	maskPoint.y = cvRound(tempPoint.y);
	allMaskPoint[0].push_back(maskPoint);

	tempPoint = tempPointSet[a.patternSize.width - 1];
	maskPoint.x = cvRound(tempPoint.x);
	maskPoint.y = cvRound(tempPoint.y);
	allMaskPoint[0].push_back(maskPoint);

	tempPoint = tempPointSet[a.patternSize.width * a.patternSize.height - 1];
	maskPoint.x = cvRound(tempPoint.x);
	maskPoint.y = cvRound(tempPoint.y);
	allMaskPoint[0].push_back(maskPoint);

	tempPoint = tempPointSet[a.patternSize.width * (a.patternSize.height - 1)];
	maskPoint.x = cvRound(tempPoint.x);
	maskPoint.y = cvRound(tempPoint.y);
	allMaskPoint[0].push_back(maskPoint);

	drawContours(mask, allMaskPoint, 0, cv::Scalar(255), cv::FILLED, 8);
	diff.copyTo(outputImage, mask);
	imwrite("./save/mask.jpg", mask);
	imwrite("./save/ROI.jpg", outputImage);

}

//计算摄像机坐标系下的 光条的三维坐标
void calc3DCoordination(Structlight a, const vector<cv::Point2f>& centerPoint, vector<cv::Point3f>& calcPoint) {
	constexpr double focalLength = 3.5; //摄像机的焦距
	const double dx = focalLength / a.cameraIntrinsic.at<double>(0, 0); //fx=f/dx fy=f/dy 摄像机在u轴和v轴方向上的尺度因子
	const double dy = focalLength / a.cameraIntrinsic.at<double>(1, 1);
	const double u0 = a.cameraIntrinsic.at<double>(0, 2); //主偏移点
	const double v0 = a.cameraIntrinsic.at<double>(1, 2);
	if (centerPoint.empty()) {
		cout << "No lightline in the image! " << endl;
		return;
	}
	//遍历一边提取出来的光条的二维点
	for (const auto& imgPoint : centerPoint) {
		cv::Point3f ccsPoint1;
		//cout << "Image Point:" << imgPoint << endl << endl;//************
		// 计算 摄像机坐标系下，图像上光条中心点，以厘米为单位的 三维坐标
		ccsPoint1.x = dx * imgPoint.x - u0 * dx;
		ccsPoint1.y = dy * imgPoint.y - v0 * dy;
		ccsPoint1.z = focalLength;

		// 射线和光平面的交点，计算光条点的世界坐标系
		cv::Point3f worldPoint(0, 0, 0);
		worldPoint.x = a.lightPlaneFormular[3] / (a.lightPlaneFormular[0]
			+ a.lightPlaneFormular[1] * ccsPoint1.y / ccsPoint1.x
			+ a.lightPlaneFormular[2] * ccsPoint1.z / ccsPoint1.x);
		worldPoint.y = ccsPoint1.y * worldPoint.x / ccsPoint1.x; // x 求出来之后 y,z可以利用比例求出
		worldPoint.z = ccsPoint1.z * worldPoint.x / ccsPoint1.x;

		calcPoint.push_back(worldPoint);
	}
	//// 将三维坐标写入TXT文件中
	//ofstream outfile;
	//outfile.open("pointcloud.txt", ios::binary | ios::in | ios::out);
	//for (int k = 0; k < calcPoint.size(); k++)
	//{
	//	outfile << calcPoint[k].x << "   ";
	//	outfile << calcPoint[k].y << "   ";
	//	outfile << calcPoint[k].z << "\n";
	//}
	//outfile.close();
}

/* 误差评估函数，思路是计算标定用的7个点和点云之间的平均距离，进一步评估测量效果 */
void estimateError(const structure_light& structureLightCalib, const Structlight& structLightMeasure,
                   vector<vector<float>> backProjectError) {
	/* 评估之前定义靶标的列数和评估的图像编号 */
	constexpr int boardRow = 7;
	constexpr int imageNum = 2;
	vector<float> backProjectErrorTemp;
	//光条和每列圆斑的交点的三维坐标（所有图的）
	const vector<cv::Point3f> crossPoint = structureLightCalib.lightPlanePoint;
	/* 用拟合出来的平面计算光条中心的三维坐标 */
	vector<cv::Point3f> calcPoint; //第ImageNum张图光条中心的三维坐标
	calc3DCoordination(structLightMeasure, structureLightCalib.lightPlaneSubPixelImagePoint[imageNum], calcPoint);
	cv::Mat imageTemp = cv::imread("./light_picture/light4.jpg");
	for (auto& ii : structureLightCalib.lightPlaneSubPixelImagePoint[imageNum])
		circle(imageTemp, ii, 1, cv::Scalar(0, 255, 0));
	imwrite("./save/imagetemp.jpg", imageTemp);
	/* 计算点之间的最小距离 评估误差 */
	for (int i = imageNum * boardRow; i < (imageNum + 1) * boardRow; i++) {
		float minDistance = sqrt((crossPoint[i].x - calcPoint[0].x) * (crossPoint[i].x - calcPoint[0].x)
			+ (crossPoint[i].y - calcPoint[0].y) * (crossPoint[i].y - calcPoint[0].y)
			+ (crossPoint[i].z - calcPoint[0].z) * (crossPoint[i].z - calcPoint[0].z));

		for (const auto& k : calcPoint) {
			const float distance = sqrt((crossPoint[i].x - k.x) * (crossPoint[i].x - k.x)
				+ (crossPoint[i].y - k.y) * (crossPoint[i].y - k.y)
				+ (crossPoint[i].z - k.z) * (crossPoint[i].z - k.z));
			if (minDistance > distance)
				minDistance = distance;
			else
				minDistance = minDistance;
		}
		backProjectErrorTemp.push_back(minDistance);
		cout << "最小距离" << minDistance << endl;
	}
	backProjectError.push_back(backProjectErrorTemp);

	// 将三维坐标写入TXT文件中
	ofstream outfile;
	outfile.open("pointCloud2.txt", ios::binary | ios::in | ios::out);
	for (int k = imageNum * boardRow; k < (imageNum + 1) * boardRow; k++) {
		outfile << crossPoint[k].x << "          ";
		outfile << crossPoint[k].y << "          ";
		outfile << crossPoint[k].z << "\n";
	}
	for (const auto& k : calcPoint) {
		outfile << k.x << "          ";
		outfile << k.y << "          ";
		outfile << k.z << "\n";
	}
	outfile.close();

}

/* 误差评估函数，思路是计算标定用的7个点和点云拟合出的直线之间的平均距离，进一步评估测量效果 */
void estimateError2(structure_light structureLightCalib, const Structlight& structLightMeasure,
                    vector<vector<float>> backProjectError) {
	/* 评估之前定义靶标的列数和评估的图像编号 */
	int boardRow = 7;
	int imageNum = 4;
	/* 首先取出光条和每列圆斑的交点的三维坐标，这是标定（拟合平面）用的 */
	vector<cv::Point3f> crossPoint; //光条和每列圆斑的交点的三维坐标（所有图的）
	crossPoint = structureLightCalib.lightPlanePoint;
	/* 用拟合出来的平面计算光条中心的三维坐标 */
	vector<cv::Point3f> calcPoint; //第ImageNum张图光条中心的三维坐标
	calc3DCoordination(structLightMeasure, structureLightCalib.lightPlaneSubPixelImagePoint[imageNum], calcPoint);
	cv::Mat imageTemp = cv::imread("./light_picture/light4.jpg");
	for (auto& ii : structureLightCalib.lightPlaneSubPixelImagePoint[imageNum])
		circle(imageTemp, ii, 1, cv::Scalar(0, 255, 0));
	imwrite("./save/imagetemp.jpg", imageTemp);
	/*################# 对重构的点云进行直线拟合 #################*/
	/*把坐标转换成矩阵 N*3的矩阵 */
	vector<vector<double>> aPointCloud;
	for (auto& i : calcPoint) {
		vector<double> aPointCloudTemp;
		aPointCloudTemp.push_back(i.x);
		aPointCloudTemp.push_back(i.y);
		aPointCloudTemp.push_back(i.z);
		aPointCloud.push_back(aPointCloudTemp);
	}
	cout << aPointCloud.size() << endl;
	/* 拟合的直线必过所有坐标的算数平均值 得到直线上一点 */
	double lineX0, lineY0, lineZ0, xSum = 0, ySum = 0, zSum = 0;
	for (auto& i : aPointCloud) {
		xSum = xSum + i[0];
		ySum = ySum + i[1];
		zSum = zSum + i[2];
	}
	lineX0 = xSum / aPointCloud.size();
	lineY0 = ySum / aPointCloud.size();
	lineZ0 = zSum / aPointCloud.size(); //平均值就是直线经过的点

	/* 协方差矩阵奇异变换（SVD分解）得到直线方向向量 */
	vector<vector<double>> U;
	vector<double> S;
	vector<vector<double>> V;
	for (auto& i : aPointCloud) {
		i[0] = i[0] - lineX0;
		i[1] = i[1] - lineY0;
		i[2] = i[2] - lineZ0;
	}
	svd(aPointCloud, 1, U, S, V);
	cout << "V矩阵大小 " << V.size() << " [ " << V[0][0] << " " << V[0][1] << " " << V[0][2] << " ] " << endl;
	//直线方程 (x - x0)/m = (y - y0)/n = (z - z0)/p = t ;x0,y0,z0是LineX0，LineY0，LineZ0；m,n,p对应这V的三个元素
	/*################# 拟合完成 #################*/
	/* 计算点之间的最小距离 评估误差 */
	vector<float> backProjectErrorTemp;
	for (int i = imageNum * boardRow; i < (imageNum + 1) * boardRow; i++) {
		float MinDistacne = sqrt((crossPoint[i].x - calcPoint[0].x) * (crossPoint[i].x - calcPoint[0].x)
			+ (crossPoint[i].y - calcPoint[0].y) * (crossPoint[i].y - calcPoint[0].y)
			+ (crossPoint[i].z - calcPoint[0].z) * (crossPoint[i].z - calcPoint[0].z));
		float t, Xc, Yc, Zc; //t直线方程的参数，Xc,Yc,Zc是标定用的点做直线的垂线的垂足；下边是计算公式，目的是计算标定用点到拟合直线的距离
		t = -(V[0][0] * (lineX0 - crossPoint[i].x) + V[0][1] * (lineY0 - crossPoint[i].y) + V[0][2] * (lineZ0 -
			crossPoint[i].z)) / (V[0][0] * V[0][0] + V[0][1] * V[0][1] + V[0][2] * V[0][2]);
		Xc = V[0][0] * t + lineX0;
		Yc = V[0][1] * t + lineY0;
		Zc = V[0][2] * t + lineZ0;
		float distance = sqrt((crossPoint[i].x - Xc) * (crossPoint[i].x - Xc)
			+ (crossPoint[i].y - Yc) * (crossPoint[i].y - Yc)
			+ (crossPoint[i].z - Zc) * (crossPoint[i].z - Zc));
		backProjectErrorTemp.push_back(distance);
		cout << "点到线的距离" << distance << endl;
	}
	/* 计算7个点的平均距离 */
	float distanceSum = 0, DistanceMaen = 0;
	for (float i : backProjectErrorTemp)
		distanceSum = distanceSum + i;
	DistanceMaen = distanceSum / backProjectErrorTemp.size();
	cout << "平均距离: " << DistanceMaen << endl;
	backProjectError.push_back(backProjectErrorTemp);

	// 将三维坐标写入TXT文件中
	ofstream outfile;
	outfile.open("pointcloud2.txt", ios::binary | ios::in | ios::out);
	for (int k = imageNum * boardRow; k < (imageNum + 1) * boardRow; k++) {
		outfile << crossPoint[k].x << "          ";
		outfile << crossPoint[k].y << "          ";
		outfile << crossPoint[k].z << "\n";
	}
	for (auto& k : calcPoint) {
		outfile << k.x << "          ";
		outfile << k.y << "          ";
		outfile << k.z << "\n";
	}
	outfile.close();

}

/* ############ SVD相关代码 ########################*/
double get_norm(double* x, int n) {
	double r = 0;
	for (int i = 0; i < n; i++)
		r += x[i] * x[i];
	return sqrt(r);
}

double normalize(double* x, int n) {
	const double r = get_norm(x, n);
	if (r < eps)
		return 0;
	for (int i = 0; i < n; i++)
		x[i] /= r;
	return r;
}

inline double product(double* a, double* b, int n) {
	double r = 0;
	for (int i = 0; i < n; i++)
		r += a[i] * b[i];
	return r;
}

void orth(double* a, double* b, int n) {
	//|a|=1
	const double r = product(a, b, n);
	for (int i = 0; i < n; i++)
		b[i] -= r * a[i];

}

bool svd(vector<vector<double>> A, int K, vector<vector<double>>& U, vector<double>& S, vector<vector<double>>& V) {
	const int m = A.size();
	const int n = A[0].size();
	U.clear();
	V.clear();
	S.clear();
	S.resize(K, 0);
	U.resize(K);
	for (int i = 0; i < K; i++)
		U[i].resize(m, 0);
	V.resize(K);
	for (int i = 0; i < K; i++)
		V[i].resize(n, 0);


	srand(time(nullptr));
	const auto leftVector = new double[m];
	const auto nextLeftVector = new double[m];
	const auto rightVector = new double[n];
	const auto nextRightVector = new double[n];
	for (int col = 0; col < K; col++) {
		double diff = 1;
		double r = -1;
		while (true) {
			for (int i = 0; i < m; i++)
				leftVector[i] = static_cast<float>(rand()) / RAND_MAX;
			if (normalize(leftVector, m) > eps)
				break;
		}

		for (int iter = 0; diff >= eps && iter < MAX_ITER; iter++) {
			memset(nextLeftVector, 0, sizeof(double) * m);
			memset(nextRightVector, 0, sizeof(double) * n);
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					nextRightVector[j] += leftVector[i] * A[i][j];

			r = normalize(nextRightVector, n);
			if (r < eps) break;
			for (int i = 0; i < col; i++)
				orth(&V[i][0], nextRightVector, n);
			normalize(nextRightVector, n);

			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					nextLeftVector[i] += nextRightVector[j] * A[i][j];
			r = normalize(nextLeftVector, m);
			if (r < eps) break;
			for (int i = 0; i < col; i++)
				orth(&U[i][0], nextLeftVector, m);
			normalize(nextLeftVector, m);
			diff = 0;
			for (int i = 0; i < m; i++) {
				double d = nextLeftVector[i] - leftVector[i];
				diff += d * d;
			}

			memcpy(leftVector, nextLeftVector, sizeof(double) * m);
			memcpy(rightVector, nextRightVector, sizeof(double) * n);
		}
		if (r >= eps) {
			S[col] = r;
			memcpy(&U[col][0], leftVector, sizeof(double) * m);
			memcpy(&V[col][0], rightVector, sizeof(double) * n);
		}
		else {
			std::cout << r << std::endl;
			break;
		}
	}
	delete[] nextLeftVector;
	delete[] nextRightVector;
	delete[] leftVector;
	delete[] rightVector;

	return true;
}

/* ############ SVD相关代码结束 ########################*/
