#include <utility>

#include "calibration.h"

using namespace std;

ofstream file("caliberation_result.txt"); //保存标定结果的文件)

CalibrateCamera::CalibrateCamera() = default;

CalibrateCamera::~CalibrateCamera() = default;

void CalibrateCamera::LoadCalibImage(const vector<string>& imgPathList) {
	vector<cv::Point2f> cornerPointsBuf; //建一个数组缓存检测到的角点，通常采用Point2f形式,以图像的左上角为原点，而不是以棋盘的左上角为原点
	vector<cv::Point2f>::iterator cornerPointsBufPtr;
	vector<vector<cv::Point2f>> cornerPointsOfAllImages; //所有图片的角点信息
	for (int i = 0; i < imgPathList.size(); ++i) {
		cv::Mat imageInput = cv::imread(imgPathList.at(i));
		if (i == 1) {
			imageSize_.width = imageInput.cols;
			imageSize_.height = imageInput.rows;
			cout << "image_size.width = " << imageSize_.width << endl;
			cout << "image_size.height = " << imageSize_.height << endl;
		}
		imgList_.push_back(imageInput);
	}
}

void CalibrateCamera::setPatternSize(cv::Size size) {
	patternSize_ = std::move(size);
}

// 提取棋盘格角点
void CalibrateCamera::cornerPointExtraction() {
	vector<cv::Point2f> cornerPointsBuf; //建一个数组缓存检测到的角点，通常采用Point2f形式,以图像的左上角为原点，而不是以棋盘的左上角为原点
	vector<cv::Point2f>::iterator cornerPointsBufPtr;
	for (auto& i : imgList_) {
		if (findChessboardCorners(i, patternSize_, cornerPointsBuf) == 0) {
			std::cout << "Can not find all chessboard corners!\n"; //找不到角点
			return;
		}
		cv::Mat gray;
		if (i.channels() == 3)
			cvtColor(i, gray, cv::COLOR_RGB2GRAY); //将原来的图片转换为灰度图片
		else
			gray = i;
		find4QuadCornerSubpix(gray, cornerPointsBuf, cv::Size(5, 5)); //提取亚像素角点,Size(5, 5),角点搜索窗口的大小。
		cornerPointsOfAllImages_.push_back(cornerPointsBuf);
		drawChessboardCorners(gray, patternSize_, cornerPointsBuf, true);
		cv::namedWindow("camera calibration", 0);
		cv::resizeWindow("camera calibration", 960, 640);
		imshow("camera calibration", gray);
		cv::waitKey(50);
	}

	const auto total = cornerPointsOfAllImages_.size();
	cout << "total=" << total << endl;
	const int cornerNum = patternSize_.width * patternSize_.height; //每张图片上的总的角点数
	for (int i = 0; i < total; i++) {
		cout << "--> 第" << i + 1 << "幅图片的数据 -->:" << endl;
		for (int j = 0; j < cornerNum; j++) {
			cout << "-->" << cornerPointsOfAllImages_[i][j].x;
			cout << "-->" << cornerPointsOfAllImages_[i][j].y;
			if ((j + 1) % 3 == 0)
				cout << endl;
			else
				cout.width(10);
		}
		cout << endl;
	}

	cout << endl << "角点提取完成" << endl;
	cv::destroyAllWindows();
}

void CalibrateCamera::cameraCalibration() {
	cameraMatrix_ = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); //内外参矩阵，H——单应性矩阵
	distCoefficients_ = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0)); //摄像机的5个畸变系数：k1,k2,p1,p2,k3
	//初始化每一张图片中标定板上角点的三维坐标
	//世界坐标系，以棋盘格的左上角为坐标原点
	for (int k = 0; k < imgList_.size(); k++) {
		vector<cv::Point3f> tempCornerPoints; //每一幅图片对应的角点数组
		//遍历所有的角点
		for (int i = 0; i < patternSize_.height; i++)
			for (int j = 0; j < patternSize_.width; j++) {
				cv::Point3f singleRealPoint; //一个角点的坐标，初始化三维坐标
				singleRealPoint.x = i * 5; //10是长/宽，根据黑白格子的长和宽，计算出世界坐标系(x,y,z)
				singleRealPoint.y = j * 5;
				singleRealPoint.z = 0; //假设z=0
				tempCornerPoints.push_back(singleRealPoint);
			}
		objectPoints_.push_back(tempCornerPoints);
	}

	calibrateCamera(objectPoints_, cornerPointsOfAllImages_, imageSize_, cameraMatrix_,
		distCoefficients_, rvecsMat, tvecsMat, 0);

	cout << endl << "相机相关参数：" << endl;
	file << "相机相关参数：" << endl;
	cout << "1.内外参数矩阵:" << endl;
	file << "1.内外参数矩阵:" << endl;
	cout << "大小：" << cameraMatrix_.size() << endl;
	file << "大小：" << cameraMatrix_.size() << endl;
	cout << cameraMatrix_ << endl;
	file << cameraMatrix_ << endl;
	cout << "大小：" << distCoefficients_.size() << endl;
	file << "大小：" << distCoefficients_.size() << endl;
	cout << distCoefficients_ << endl;
	file << distCoefficients_ << endl;
}

void CalibrateCamera::calculateError() {
	const int cornerPointsCounts = patternSize_.width * patternSize_.height;
	totalErr_ = 0;
	cout << "每幅图像的标定误差：" << endl;
	file << "每幅图像的标定误差：" << endl;
	for (int i = 0; i < imgList_.size(); i++) {
		vector<cv::Point2f> imagePointsCalculated; //存放新计算出的投影点的坐标
		vector<cv::Point3f> tempPointSet = objectPoints_[i];
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix_, distCoefficients_, imagePointsCalculated);
		//计算根据内外参等重投影出来的新的二维坐标，输出到image_points_calculated
		//计算新的投影点与旧的投影点之间的误差
		vector<cv::Point2f> imagePointsOld = cornerPointsOfAllImages_[i]; //向量
		//将两组数据换成Mat格式
		auto imagePointsCalculatedMat = cv::Mat(1, imagePointsCalculated.size(), CV_32FC2); //将mat矩阵转成1维的向量
		auto imagePointsOldMat = cv::Mat(1, imagePointsOld.size(), CV_32FC2);
		for (int j = 0; j < tempPointSet.size(); j++) {
			imagePointsCalculatedMat.at<cv::Vec2f>(0, j) =
				cv::Vec2f(imagePointsCalculated[j].x, imagePointsCalculated[j].y); //vec2f->一个二维的向量
			imagePointsOldMat.at<cv::Vec2f>(0, j) = cv::Vec2f(imagePointsOld[j].x, imagePointsOld[j].y); //直接调用函数，不用定义对象
		}
		double err = cv::norm(imagePointsCalculatedMat, imagePointsOldMat, cv::NORM_L2); //输入的是矩阵
		err /= cornerPointsCounts; //每个角点的误差
		totalErr_ += err;
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		file << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << totalErr_ / static_cast<int>(imgList_.size()) << "像素" << endl;
	file << "总体平均误差：" << totalErr_ / static_cast<int>(imgList_.size()) << "像素" << endl;
	cout << "评价完成" << endl;
}

void CalibrateCamera::adjustImage(const cv::Mat& image) const {
	auto mapX = cv::Mat(imageSize_, CV_32FC1); //对应坐标的重映射参数
	auto mapY = cv::Mat(imageSize_, CV_32FC1);
	const cv::Mat r = cv::Mat::eye(3, 3, CV_32F); //定义的对角线为1的对角矩阵
	cout << "保存矫正图像" << endl;
	string imageFileName;
	initUndistortRectifyMap(cameraMatrix_, distCoefficients_, r, cameraMatrix_, imageSize_, CV_32FC1, mapX, mapY);
	//输入内参，纠正后的内参，外参等,计算输出矫正的重映射参数(相片尺寸 width*height),每个像素点都需要转换
	cv::Mat newImage = image.clone();
	remap(image, newImage, mapX, mapY, cv::INTER_LINEAR);
	cv::namedWindow("原始图像", 0);
	cv::resizeWindow("原始图像", 960, 640);
	cv::namedWindow("矫正后图像", 0);
	cv::resizeWindow("矫正后图像", 960, 640);
	imshow("原始图像", image);
	imshow("矫正后图像", newImage);

	cv::waitKey();
	cv::destroyAllWindows();
	file.close(); //
}

void CalibrateCamera::run() {
	if(!imgList_.empty()) {
		cornerPointExtraction();
		cameraCalibration();
	}
}
