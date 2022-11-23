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

//������궨
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
		//Բ�ΰб���ȡԲ�ĵ�
		//��ȡ�б���Բ�ߵ�Բ��
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
		//���̸�б���ȡ�ǵ�
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
	/* ����ͷ�궨�õ�����ڲκͻ�������Լ���Σ���ת����R��ƽ�ƾ���T��*/
	double rms = calibrateCamera(a.calibBoardPoint, a.calibImagePoint, imageSize, a.cameraMatrix, a.distCoeffs, a.R,
	                             a.T);
	cout << "Re-projection error:" << rms << endl;

	a.Rw = a.R[0];
	a.Tw = a.T[0];
	/* ��ͶӰ������Ŀ����ͷ�궨���� */
	double err = 0.0;
	double meanErr = 0.0;
	double totalErr = 0.0;
	int i;
	vector<cv::Point2f> reProjectionPoint;

	for (i = 0; i < a.calibBoardPoint.size(); i++) {
		vector<cv::Point3f> tempPointSet = a.calibBoardPoint[i];
		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		projectPoints(tempPointSet, a.R[i], a.T[i], a.cameraMatrix, a.distCoeffs, reProjectionPoint);
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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

//################# RANSAC�㷨���ֱ����غ��� #########################//
//����[0,1]֮����Ͼ��ȷֲ�����
double uniformRandom() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }

//ֱ���������������λ�ò���̫��
bool verifyComposition(const vector<cv::Point2f>& pts) {
	const cv::Point2d pt1 = pts[0];
	const cv::Point2d pt2 = pts[1];
	if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
		return false;
	return true;
}

//���ݵ㼯���ֱ��ax+by+c=0��resΪ�в�
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

//�õ�ֱ���������������ֱ�߲����㼯�����ѡ2����
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


//RANSACֱ�����
void fitLineRANSAC(vector<cv::Point2f> ptSet, double& a, double& b, double& c, vector<bool>& inlierFlag) {

	bool stopLoop = false;
	int maximum = 0; //����ڵ���

	//�����ڵ��ʶ����в�
	inlierFlag = vector<bool>(ptSet.size(), false);
	vector<double> resids_(ptSet.size(), 3);
	int sampleCount = 0;
	int N = 500;

	double res = 0;

	// RANSAC
	srand(static_cast<unsigned>(time(nullptr))); //�������������
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

		// ����ֱ�߷���
		calcLinePara(pt_sam, a, b, c, res);
		//�ڵ����
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
		// �ҵ�������ֱ��
		if (inlierCount >= maximum) {
			maximum = inlierCount;
			resids_ = residualStemp;
			inlierFlag = inlierstemp;
		}
		// ����RANSAC�����������Լ��ڵ����
		if (inlierCount == 0)
			N = 500;
		else {
			const double epsilon = 1.0 - static_cast<double>(inlierCount) / static_cast<double>(ptSet.size()); //Ұֵ�����
			constexpr double p = 0.99; //���������д���1���������ĸ���
			constexpr double s = 2.0;
			N = static_cast<int>(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
		}
		++sampleCount;
	}

	//���������ڵ��������ֱ��
	vector<cv::Point2f> pset;
	for (unsigned int i = 0; i < ptSet.size(); i++)
		if (inlierFlag[i])
			pset.push_back(ptSet[i]);

	calcLinePara(pset, a, b, c, res);
}

//##################### RANSAC�㷨���ֱ�߲������ ######################


//��ȡ�ṹ��������ĵ㣨steger�㷨��
void stegerLine(structure_light& a) {
	for (int k = 0; k < a.imageCount; k++) {
		if (a.isRecognize[k] == false)
			continue;

		cv::Mat lightLineImage;
		extratLightLine(k, lightLineImage, a);

		imwrite("./save/image.jpg", lightLineImage);
		//һ��ƫ����
		cv::Mat m1, m2;
		m1 = (cv::Mat_<float>(1, 2) << 1, -1);
		m2 = (cv::Mat_<float>(2, 1) << 1, -1);

		cv::Mat dx, dy;
		filter2D(lightLineImage, dx, CV_32FC1, m1);
		filter2D(lightLineImage, dy, CV_32FC1, m2);

		//����ƫ����
		cv::Mat m3, m4, m5;
		m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1); //����xƫ��
		m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1); //����yƫ��
		m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1); //����xyƫ��

		cv::Mat dxx, dyy, dxy;
		filter2D(lightLineImage, dxx, CV_32FC1, m3);
		filter2D(lightLineImage, dyy, CV_32FC1, m4);
		filter2D(lightLineImage, dxy, CV_32FC1, m5);

		vector<cv::Point2f> oriImagePoint;
		vector<cv::Point2f> subPixelImagePoint;
		//hessian������ȡ��������
		for (int i = 0; i < lightLineImage.cols; i++)
			for (int j = 0; j < lightLineImage.rows; j++)
				//ͨ���Ҷ�ֵȷ����������

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
						subPixelImagePoint.push_back(subPixelPoint); //�����صĹ������ĵ�
					}
				}
		a.lightPlaneOriImagePoint.push_back(oriImagePoint);
		//a.lightPlaneSubPixelImagePoint.push_back(subPixelImagePoint);
		/* RANSAC�㷨���ֱ�ߣ��˳����� */
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

		//����ֱ��
		//Point2d ptStart, ptEnd;
		//ptStart.x = 0;
		//ptStart.y = -(A*ptStart.x + C) / B;
		//ptEnd.x = -(B*ptEnd.y + C) / A;
		//ptEnd.y = 0;
		//line(ImageRANSAC, ptStart, ptEnd, Scalar(0, 0, 0), 1, 16);
		//cout << "A:" << A << " " << "B:" << B << " " << "C:" << C << " " << endl;
		//for (int k = 0; k < subPixelImagePoint.size(); k++)//��ͼ�л���Щ�㣬Ȼ�󱣴�ͼƬ
		//{
		//	circle(ImageRANSAC, subPixelImagePoint[k], 1, Scalar(0, 255, 0));
		//}
		//imwrite(".\\save\\imageRANSAC.jpg", ImageRANSAC);
		//�������е㵽���ֱ�ߵľ���
		//Mat ImageNO = imread(".\\light_picture\\light1.jpg");
		vector<cv::Point2f> subPixelImagePointTemp;
		for (auto& i : subPixelImagePoint) {
			float distance = abs(A * i.x + B * i.y + C) / sqrt(A * A + B * B);
			//cout << "��" << i << "��ֱ�ߵľ��� :  "<<  distance  << endl;
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

		/* RANSAC�㷨���ֱ�߲��ֽ��� */

		cv::Mat lightLineImageResult = cv::Mat::zeros(lightLineImage.rows, lightLineImage.cols, CV_8U);
		for (auto& k : subPixelImagePoint) //��ͼ�л���Щ�㣬Ȼ�󱣴�ͼƬ
			circle(lightLineImageResult, k, 1, cv::Scalar(255, 255, 255));
		//float distance1 = abs(A*subPixelImagePoint[k].x + B * subPixelImagePoint[k].y + C) / sqrt(A*A + B * B);
		//cout << "��" << k << "��ֱ�ߵľ��� :  " << distance1 << endl;
		string format = ".jpg";
		string index = to_string(k);
		string name = "./save/result" + index + format;
		imwrite(name, lightLineImageResult);
		oriImagePoint.clear();
		subPixelImagePoint.clear();
	}
}

//�ṹ�����ֱ��������ཻ��������ȡ
//����ṹ����͵�vector ����֮��vector���ǹ�ƽ��Ͱб�ƽ���ཻ��ֱ���ϵĵ㣨7*7 = 49�� ��
void crossPoint(const structure_light& a, vector<vector<cv::Point2f>>& crossPoint) {
	for (int i = 0; i < imageCount; i++) {
		if (a.isRecognize[i] == false)
			continue;
		cv::Vec4f lightLine;
		vector<cv::Point2f> lightLinePoint = a.lightPlaneSubPixelImagePoint[i]; //ȡ����i�ű궨ͼ�� �������ع������ĵ�
		fitLine(lightLinePoint, lightLine, cv::DIST_L2, 0, 1e-2, 1e-2); //����i��ͼ�� �������ĵ����ֱ�����
		vector<cv::Point2f> cornerPoint = a.calibImagePoint[i]; //�б�ƽ����Բ�α�����ĵĵ㣨49�����Ķ�ά����
		vector<cv::Point2f> cornerLinePoint; //�洢ĳһ�е�Բ�α�ǵ�����
		cv::Vec4f cornerLine; //һ��Բ���������ϳ���ֱ�ߵĲ���
		vector<cv::Point2f> lightCornerPoint; //��ȡ���Ĺ����ϵĵ�
		//��ȡ�����ϵĵ㣬˼·��ȡһ�У�����һ�У�Բ�α�ǵ����ģ���ϳ�ֱ�ߣ�������ϳ�ֱ�ߺ͹����Ľ���
		//����������7��
		for (int m = 0; m < patternSize.width; m++) {
			for (int n = 0; n < patternSize.height; n++)
				cornerLinePoint.push_back(cornerPoint[n * patternSize.width + m]); //ȡ��m��Բ�α�ǵ�����
			fitLine(cornerLinePoint, cornerLine, cv::DIST_L2, 0, 1e-2, 1e-2); //����m�е�Բ����ϳ�ֱ��
			//�����m��Բ����ϳ�����ֱ�ߺ͹���ֱ�ߵĽ���
			const double k1 = cornerLine[1] / cornerLine[0];
			const double b1 = cornerLine[3] - k1 * cornerLine[2];
			const double k2 = lightLine[1] / lightLine[0];
			const double b2 = lightLine[3] - k2 * lightLine[2];
			cv::Point2f temp;
			temp.x = (b2 - b1) / (k1 - k2);
			temp.y = k1 * temp.x + b1;
			//������ѹ������
			lightCornerPoint.push_back(temp);
			cornerLinePoint.clear();
		}
		crossPoint.push_back(lightCornerPoint);
	}
}

//���ý��Ȳ�����ȡ��ƽ����ά����
void crossRatio(structure_light& light, vector<vector<cv::Point2f>>& crossPoint) {
	for (int i = 0; i < imageCount; i++) {
		if (light.isRecognize[i] == false)
			continue;
		vector<cv::Point2f> tempCrossPoint = crossPoint[i]; //ĳһ��ͼ�еĹ����ϵ�7����Ķ�ά����
		vector<cv::Point2f> tempCornerPoint = light.calibImagePoint[i]; //��i���б�ƽ����Բ�α�����ĵĵ㣨49�����Ķ�ά����
		vector<cv::Point3f> tempWorldPoint = light.calibBoardPoint[i]; //��i���б�ƽ����Բ�α�����ĵĵ㣨49��������ά������
		vector<cv::Point3f> tempCornerLinePoint; //���ڴ�����ϵĵ����ά����
		//�����i���б��ϵ�7�������ϵĵ����ά���꣨�б�����ϵ�£�ѭ��7��
		for (int m = 0; m < tempCrossPoint.size(); m++) {
			//��ȡÿ��Բ�ߵ��γ�ֱ����ǰ���������������
			cv::Point2f a, b, c;
			cv::Point3f A, B, C;
			a = tempCornerPoint[m];
			b = tempCornerPoint[patternSize.width + m];
			c = tempCornerPoint[2 * patternSize.width + m];
			A = tempWorldPoint[m];
			B = tempWorldPoint[patternSize.width + m];
			C = tempWorldPoint[2 * patternSize.width + m];

			//���㽻��
			double crossRatio = ((a.y - c.y) / (b.y - c.y)) / ((a.y - tempCrossPoint[m].y) / (b.y - tempCrossPoint[m].
				y));
			//double crossRatio = ((a.x - c.x) / (b.x - c.x)) / ((a.x - tempCrossPoint[m].x) / (b.x - tempCrossPoint[m].x));
			//��֪4�����ͼ��������3�����������꣬���Լ�������1����������꣨������֪����Բ�α�����ĺ�һ�������ϵĵ��ͼ�����꣬��������ǵ����ά���꣩
			cv::Point3f crPoint;
			crPoint.x = m * patternLength.width;
			crPoint.y = (crossRatio * (B.y - C.y) * A.y - B.y * (A.y - C.y)) / (crossRatio * (B.y - C.y) - (A.y - C.y));
			//���Y����
			crPoint.z = 0;
			//cout << "ccs Point:" << crPoint << endl;
			tempCornerLinePoint.push_back(crPoint);
		}

		cv::Mat rvec = light.R[i];
		cv::Mat T1 = light.T[i]; //�б�����ϵ�����������ϵ֮�����תƽ�ƾ���
		cv::Mat rMat1, rMat2;
		Rodrigues(rvec, rMat1); //1x3 -> 3x3 ��ת������1x3������ת����3x3�������߿���ͨ���޵����˹�໥ת��

		//���б�����ϵ�µ���ά����ת�������������ϵ�µ���ά����
		vector<cv::Point3f> tempPoint;
		for (auto& n : tempCornerLinePoint) {
			cv::Point3f realPoint; //���������ϵ�µ����꣬������������ת�任��������ת��
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

//�ṹ��Ĺ�ƽ�����
void lightPlainFitting(structure_light& light) {
	cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1); //�����������Ҫ����������
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
	light.planeFormular[3] = X.at<double>(2, 0); //������
}

//��������ȡ
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

//�������������ϵ�µ� ��������ά����
void calc3DCoordination(Structlight a, const vector<cv::Point2f>& centerPoint, vector<cv::Point3f>& calcPoint) {
	constexpr double focalLength = 3.5; //������Ľ���
	const double dx = focalLength / a.cameraIntrinsic.at<double>(0, 0); //fx=f/dx fy=f/dy �������u���v�᷽���ϵĳ߶�����
	const double dy = focalLength / a.cameraIntrinsic.at<double>(1, 1);
	const double u0 = a.cameraIntrinsic.at<double>(0, 2); //��ƫ�Ƶ�
	const double v0 = a.cameraIntrinsic.at<double>(1, 2);
	if (centerPoint.empty()) {
		cout << "No lightline in the image! " << endl;
		return;
	}
	//����һ����ȡ�����Ĺ����Ķ�ά��
	for (const auto& imgPoint : centerPoint) {
		cv::Point3f ccsPoint1;
		//cout << "Image Point:" << imgPoint << endl << endl;//************
		// ���� ���������ϵ�£�ͼ���Ϲ������ĵ㣬������Ϊ��λ�� ��ά����
		ccsPoint1.x = dx * imgPoint.x - u0 * dx;
		ccsPoint1.y = dy * imgPoint.y - v0 * dy;
		ccsPoint1.z = focalLength;

		// ���ߺ͹�ƽ��Ľ��㣬������������������ϵ
		cv::Point3f worldPoint(0, 0, 0);
		worldPoint.x = a.lightPlaneFormular[3] / (a.lightPlaneFormular[0]
			+ a.lightPlaneFormular[1] * ccsPoint1.y / ccsPoint1.x
			+ a.lightPlaneFormular[2] * ccsPoint1.z / ccsPoint1.x);
		worldPoint.y = ccsPoint1.y * worldPoint.x / ccsPoint1.x; // x �����֮�� y,z�������ñ������
		worldPoint.z = ccsPoint1.z * worldPoint.x / ccsPoint1.x;

		calcPoint.push_back(worldPoint);
	}
	//// ����ά����д��TXT�ļ���
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

/* �������������˼·�Ǽ���궨�õ�7����͵���֮���ƽ�����룬��һ����������Ч�� */
void estimateError(const structure_light& structureLightCalib, const Structlight& structLightMeasure,
                   vector<vector<float>> backProjectError) {
	/* ����֮ǰ����б��������������ͼ���� */
	constexpr int boardRow = 7;
	constexpr int imageNum = 2;
	vector<float> backProjectErrorTemp;
	//������ÿ��Բ�ߵĽ������ά���꣨����ͼ�ģ�
	const vector<cv::Point3f> crossPoint = structureLightCalib.lightPlanePoint;
	/* ����ϳ�����ƽ�����������ĵ���ά���� */
	vector<cv::Point3f> calcPoint; //��ImageNum��ͼ�������ĵ���ά����
	calc3DCoordination(structLightMeasure, structureLightCalib.lightPlaneSubPixelImagePoint[imageNum], calcPoint);
	cv::Mat imageTemp = cv::imread("./light_picture/light4.jpg");
	for (auto& ii : structureLightCalib.lightPlaneSubPixelImagePoint[imageNum])
		circle(imageTemp, ii, 1, cv::Scalar(0, 255, 0));
	imwrite("./save/imagetemp.jpg", imageTemp);
	/* �����֮�����С���� ������� */
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
		cout << "��С����" << minDistance << endl;
	}
	backProjectError.push_back(backProjectErrorTemp);

	// ����ά����д��TXT�ļ���
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

/* �������������˼·�Ǽ���궨�õ�7����͵�����ϳ���ֱ��֮���ƽ�����룬��һ����������Ч�� */
void estimateError2(structure_light structureLightCalib, const Structlight& structLightMeasure,
                    vector<vector<float>> backProjectError) {
	/* ����֮ǰ����б��������������ͼ���� */
	int boardRow = 7;
	int imageNum = 4;
	/* ����ȡ��������ÿ��Բ�ߵĽ������ά���꣬���Ǳ궨�����ƽ�棩�õ� */
	vector<cv::Point3f> crossPoint; //������ÿ��Բ�ߵĽ������ά���꣨����ͼ�ģ�
	crossPoint = structureLightCalib.lightPlanePoint;
	/* ����ϳ�����ƽ�����������ĵ���ά���� */
	vector<cv::Point3f> calcPoint; //��ImageNum��ͼ�������ĵ���ά����
	calc3DCoordination(structLightMeasure, structureLightCalib.lightPlaneSubPixelImagePoint[imageNum], calcPoint);
	cv::Mat imageTemp = cv::imread("./light_picture/light4.jpg");
	for (auto& ii : structureLightCalib.lightPlaneSubPixelImagePoint[imageNum])
		circle(imageTemp, ii, 1, cv::Scalar(0, 255, 0));
	imwrite("./save/imagetemp.jpg", imageTemp);
	/*################# ���ع��ĵ��ƽ���ֱ����� #################*/
	/*������ת���ɾ��� N*3�ľ��� */
	vector<vector<double>> aPointCloud;
	for (auto& i : calcPoint) {
		vector<double> aPointCloudTemp;
		aPointCloudTemp.push_back(i.x);
		aPointCloudTemp.push_back(i.y);
		aPointCloudTemp.push_back(i.z);
		aPointCloud.push_back(aPointCloudTemp);
	}
	cout << aPointCloud.size() << endl;
	/* ��ϵ�ֱ�߱ع��������������ƽ��ֵ �õ�ֱ����һ�� */
	double lineX0, lineY0, lineZ0, xSum = 0, ySum = 0, zSum = 0;
	for (auto& i : aPointCloud) {
		xSum = xSum + i[0];
		ySum = ySum + i[1];
		zSum = zSum + i[2];
	}
	lineX0 = xSum / aPointCloud.size();
	lineY0 = ySum / aPointCloud.size();
	lineZ0 = zSum / aPointCloud.size(); //ƽ��ֵ����ֱ�߾����ĵ�

	/* Э�����������任��SVD�ֽ⣩�õ�ֱ�߷������� */
	vector<vector<double>> U;
	vector<double> S;
	vector<vector<double>> V;
	for (auto& i : aPointCloud) {
		i[0] = i[0] - lineX0;
		i[1] = i[1] - lineY0;
		i[2] = i[2] - lineZ0;
	}
	svd(aPointCloud, 1, U, S, V);
	cout << "V�����С " << V.size() << " [ " << V[0][0] << " " << V[0][1] << " " << V[0][2] << " ] " << endl;
	//ֱ�߷��� (x - x0)/m = (y - y0)/n = (z - z0)/p = t ;x0,y0,z0��LineX0��LineY0��LineZ0��m,n,p��Ӧ��V������Ԫ��
	/*################# ������ #################*/
	/* �����֮�����С���� ������� */
	vector<float> backProjectErrorTemp;
	for (int i = imageNum * boardRow; i < (imageNum + 1) * boardRow; i++) {
		float MinDistacne = sqrt((crossPoint[i].x - calcPoint[0].x) * (crossPoint[i].x - calcPoint[0].x)
			+ (crossPoint[i].y - calcPoint[0].y) * (crossPoint[i].y - calcPoint[0].y)
			+ (crossPoint[i].z - calcPoint[0].z) * (crossPoint[i].z - calcPoint[0].z));
		float t, Xc, Yc, Zc; //tֱ�߷��̵Ĳ�����Xc,Yc,Zc�Ǳ궨�õĵ���ֱ�ߵĴ��ߵĴ��㣻�±��Ǽ��㹫ʽ��Ŀ���Ǽ���궨�õ㵽���ֱ�ߵľ���
		t = -(V[0][0] * (lineX0 - crossPoint[i].x) + V[0][1] * (lineY0 - crossPoint[i].y) + V[0][2] * (lineZ0 -
			crossPoint[i].z)) / (V[0][0] * V[0][0] + V[0][1] * V[0][1] + V[0][2] * V[0][2]);
		Xc = V[0][0] * t + lineX0;
		Yc = V[0][1] * t + lineY0;
		Zc = V[0][2] * t + lineZ0;
		float distance = sqrt((crossPoint[i].x - Xc) * (crossPoint[i].x - Xc)
			+ (crossPoint[i].y - Yc) * (crossPoint[i].y - Yc)
			+ (crossPoint[i].z - Zc) * (crossPoint[i].z - Zc));
		backProjectErrorTemp.push_back(distance);
		cout << "�㵽�ߵľ���" << distance << endl;
	}
	/* ����7�����ƽ������ */
	float distanceSum = 0, DistanceMaen = 0;
	for (float i : backProjectErrorTemp)
		distanceSum = distanceSum + i;
	DistanceMaen = distanceSum / backProjectErrorTemp.size();
	cout << "ƽ������: " << DistanceMaen << endl;
	backProjectError.push_back(backProjectErrorTemp);

	// ����ά����д��TXT�ļ���
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

/* ############ SVD��ش��� ########################*/
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

/* ############ SVD��ش������ ########################*/
