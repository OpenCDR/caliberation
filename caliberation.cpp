#include <utility>

#include "calibration.h"

using namespace std;

ofstream file("caliberation_result.txt"); //保存标定结果的文件)


void CalibrateCamera::calRealPoint(std::vector<std::vector<cv::Point3f>>& obj, const int imageNum,
                                   const float squareSize, const float z) const {
	std::vector<cv::Point3f> imgPoint;
	for (int k = 0; k < imageNum; k++) {
		vector<cv::Point3f> tempCornerPoints; //每一幅图片对应的角点数组
		//遍历所有的角点
		//5是长/宽，根据黑白格子的长和宽，计算出世界坐标系(x,y,z)
		for (int i = 0; i < patternSize_.height; i++)
			for (int j = 0; j < patternSize_.width; j++) {
				cv::Point3f singleRealPoint; //一个角点的坐标，初始化三维坐标
				singleRealPoint.x = i * squareSize;
				singleRealPoint.y = j * squareSize;
				singleRealPoint.z = 0;
				//singleRealPoint.x = static_cast<float>(j) * squareSize - (static_cast<float>(patternSize_.width-1) / 2 * squareSize);
				//singleRealPoint.y = static_cast<float>(i) * squareSize - (static_cast<float>(patternSize_.height-1) / 2 * squareSize);
				//singleRealPoint.z = z; //假设z=0
				tempCornerPoints.push_back(singleRealPoint);
			}
		obj.push_back(tempCornerPoints);
	}
}


CalibrateCamera::CalibrateCamera() = default;

CalibrateCamera::~CalibrateCamera() = default;

void CalibrateCamera::loadCalibImage(const vector<string>& imgPathList) {
	vector<cv::Point2f> cornerPointsBuf; //建一个数组缓存检测到的角点，通常采用Point2f形式,以图像的左上角为原点，而不是以棋盘的左上角为原点
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
	for (auto& i : imgList_) {
		if (findChessboardCorners(i, patternSize_, cornerPointsBuf,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE) == 0) {
			std::cout << "Can not find all chessboard corners!\n"; //找不到角点
			return;
		}
		cv::Mat gray=i.clone();
		if (i.channels() == 3)
			cvtColor(i, gray, cv::COLOR_RGB2GRAY); //将原来的图片转换为灰度图片
		//find4QuadCornerSubpix(gray, cornerPointsBuf, cv::Size(5, 5)); //提取亚像素角点,Size(5, 5),角点搜索窗口的大小。
		cornerPointsOfAllImages_.push_back(cornerPointsBuf);
		drawChessboardCorners(i, patternSize_, cornerPointsBuf, true);
		//cv::namedWindow("camera calibration", 0);
		//cv::resizeWindow("camera calibration", 960, 640);
		//imshow("camera calibration", i);
		//cv::imwrite(std::to_string(cornerPointsBuf[0].x) + ".jpg", i);
		//cv::waitKey();
	}

	const auto total = cornerPointsOfAllImages_.size();
	cout << "total=" << total << endl;
	//const int cornerNum = patternSize_.width * patternSize_.height; //每张图片上的总的角点数
	//for (int i = 0; i < total; i++) {
		//cout << "--> 第" << i + 1 << "幅图片的数据 -->:" << endl;
	//	for (int j = 0; j < cornerNum; j++) {
			//cout << "-->" << cornerPointsOfAllImages_[i][j].x;
			//cout << "-->" << cornerPointsOfAllImages_[i][j].y;
	//		if ((j + 1) % 3 == 0)
	//			cout << endl;
	//		else
	//			cout.width(10);
	//	}
	//	cout << endl;
	//}

	cout << endl << "角点提取完成" << endl;
	cv::destroyAllWindows();
}

void CalibrateCamera::cameraCalibration() {
	cameraMatrix_ = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); //内外参矩阵，H——单应性矩阵
	distCoefficients_ = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0)); //摄像机的5个畸变系数：k1,k2,p1,p2,k3

	calRealPoint(objectPoints_, static_cast<int>(imgList_.size()), 5);
	calibrateCamera(objectPoints_, cornerPointsOfAllImages_, imageSize_, cameraMatrix_,
	                distCoefficients_, rvecsMat_, tvecsMat_, cv::CALIB_FIX_K3);

	cout << endl << "相机相关参数：" << endl;
	file << "相机相关参数：" << endl;
	cout << "1.内参数矩阵:" << endl;
	file << "1.内参数矩阵:" << endl;
	cout << "大小：" << cameraMatrix_.size() << endl;
	file << "大小：" << cameraMatrix_.size() << endl;
	cout << cameraMatrix_ << endl;
	file << cameraMatrix_ << endl;
	cout << "大小：" << distCoefficients_.size() << endl;
	file << "大小：" << distCoefficients_.size() << endl;
	cout << distCoefficients_ << endl;
	file << distCoefficients_ << endl;
}

void CalibrateCamera::baseCalculate(cv::Mat rgbImage) {
	//检测棋盘格标定板
	std::vector<cv::Point2f> corner;
	const bool isFind = findChessboardCorners(rgbImage, patternSize_, corner);
	if (isFind) {
		cv::Mat grayImage;
		cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);
		//find4QuadCornerSubpix(grayImage, corner, cv::Size(5, 5));
		//画出找到的交点
		drawChessboardCorners(rgbImage, patternSize_, corner, isFind);
		std::vector<std::vector<cv::Point3f>> objRealPoint;
		calRealPoint(objRealPoint, 1, 5);
		//计算视图的外参数 旋转向量和平移向量
		solvePnP(objRealPoint.at(0), corner, cameraMatrix_, distCoefficients_,
		         baseRVecsMat_, baseTVecsMat_, false, cv::SOLVEPNP_DLS);
		cv::namedWindow("BaseCalculate", 0);
		cv::resizeWindow("BaseCalculate", 960, 640);
		imshow("BaseCalculate", rgbImage);
		cv::waitKey();
	}
	cv::destroyWindow("BaseCalculate");
}


cv::Point3f CalibrateCamera::getWorldPoints(const cv::Point2f& inPoints, const cv::Mat& rvecs, const cv::Mat& tvecs) const {
	cv::Mat rotationMatrix; //3*3
	Rodrigues(rvecs, rotationMatrix);
	//激光在标定板平面上，标定板平面的世界坐标Z为0，是世界坐标
	constexpr double zConst = 0;
	const cv::Mat imagePoint = (cv::Mat_<double>(3, 1) << static_cast<double>(inPoints.x),
		static_cast<double>(inPoints.y), 1);
	//计算比例参数S
	//R-1*M-1*uvPoint*s=[X,Y,Zc]+R-1*t
	//R-1*M-1*uvPoint*
	//利用Z=0来计算s的参数，这里并不是为了得出Z的值
	//使用第三个分量计算s
	cv::Mat tempMat = rotationMatrix.inv() * cameraMatrix_.inv() * imagePoint;
	//[X,Y,Zc]+R-1*t
	cv::Mat tempMat2 = rotationMatrix.inv() * tvecs;
	double s = zConst + tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);
	//计算世界坐标
	//s*Pc=M(RPo+T)-->Po=(R-1)[s*(M-1)Pc-T]
	//Mat wcPoint = rotationMatrix.inv() * (s * this->cameraMatrix.inv() * imagePoint - tvecs);
	//计算x y的坐标
	cv::Mat wcPoint = rotationMatrix.inv() * (s * cameraMatrix_.inv() * imagePoint - tvecs);
	cv::Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
	return worldPoint;
}


cv::Point3f CalibrateCamera::myRotate(const cv::Point3f& p, cv::Vec6f line, const int k) const {
	cv::Mat res = (cv::Mat_<double>(3, 1) << p.x - line[3], p.y - line[4] - 0.05, p.z - line[5]);
	const cv::Mat rotationMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(k * PI / 160), sin(k * PI / 160), 0, -sin(k * PI / 160), cos(k * PI / 160));
	res = rotationMatrix * res;
	cv::Point3f final(res.at<double>(0, 0), res.at<double>(1, 0), res.at<double>(2, 0));
	return final;
}

// convert pixel coordinate to camera coordinate
cv::Point3f CalibrateCamera::getCameraPoints(const cv::Point2f& inPoints) const {
	// get pixel coordinate
	const cv::Mat imagePoint = (cv::Mat_<double>(3, 1) << static_cast<double>(inPoints.x), static_cast<double>(inPoints.y), 1);
	// cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	// imagePoint.at<double>(0, 0) = inPoints.x;
	// imagePoint.at<double>(1, 0) = inPoints.y;

	// compute corresponding camera coordinate
	cv::Mat caPoint = cameraMatrix_.inv() * imagePoint;
	cv::Point3f cameraPoint(caPoint.at<double>(0, 0), caPoint.at<double>(1, 0), caPoint.at<double>(2, 0));
	return cameraPoint;
}

void CalibrateCamera::calculateError() {
	const int cornerPointsCounts = patternSize_.width * patternSize_.height;
	totalErr_ = 0;
	//cout << "每幅图像的标定误差：" << endl;
	file << "每幅图像的标定误差：" << endl;
	for (int i = 0; i < imgList_.size(); i++) {
		vector<cv::Point2f> imagePointsCalculated; //存放新计算出的投影点的坐标
		vector<cv::Point3f> tempPointSet = objectPoints_[i];
		projectPoints(tempPointSet, rvecsMat_[i], tvecsMat_[i], cameraMatrix_,
		              distCoefficients_, imagePointsCalculated);
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
		//cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		file << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << totalErr_ / static_cast<int>(imgList_.size()) << "像素" << endl;
	file << "总体平均误差：" << totalErr_ / static_cast<int>(imgList_.size()) << "像素" << endl;
	//cout << "评价完成" << endl;
}

double CalibrateCamera::getAverageErr() const { return this->totalErr_ / static_cast<int>(imgList_.size()); }

void CalibrateCamera::adjustImage(const cv::Mat& image) const {
	auto mapX = cv::Mat(imageSize_, CV_32FC1); //对应坐标的重映射参数
	auto mapY = cv::Mat(imageSize_, CV_32FC1);
	const cv::Mat r = cv::Mat::eye(3, 3, CV_32F); //定义的对角线为1的对角矩阵
	cout << "保存矫正图像" << endl;
	string imageFileName;
	initUndistortRectifyMap(cameraMatrix_, distCoefficients_, r,
	                        cameraMatrix_, imageSize_, CV_32FC1, mapX, mapY);
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

cv::Mat CalibrateCamera::getBaseRvecsMat() { return baseRVecsMat_; }

cv::Mat CalibrateCamera::getBaseTvecsMat() { return baseTVecsMat_; }

cv::Mat CalibrateCamera::getCameraMatrix() {
	return cameraMatrix_;
}

cv::Mat CalibrateCamera::getDistCoeff() { return distCoefficients_; }

void CalibrateCamera::run() {
	if (!imgList_.empty()) {
		cornerPointExtraction();
		cameraCalibration();
		calculateError();
	}
}

void LaserPlane::calculateLaserPlane() {
	//最小二乘法拟合平面
	//获取cv::Mat的坐标系以纵向为x轴，横向为y轴，而cvPoint等则相反
	cv::Mat a = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Mat b = cv::Mat::zeros(3, 1, CV_64FC1);
	double x2 = 0, xiyi = 0, xi = 0, yi = 0, zixi = 0, ziyi = 0, zi = 0, y2 = 0;
	for (const auto& points3d : points3ds_) {
		x2 += static_cast<double>(points3d.x) * static_cast<double>(points3d.x);
		y2 += static_cast<double>(points3d.y) * static_cast<double>(points3d.y);
		xiyi += static_cast<double>(points3d.x) * static_cast<double>(points3d.y);
		xi += static_cast<double>(points3d.x);
		yi += static_cast<double>(points3d.y);
		zixi += static_cast<double>(points3d.z) * static_cast<double>(points3d.x);
		ziyi += static_cast<double>(points3d.z) * static_cast<double>(points3d.y);
		zi += static_cast<double>(points3d.z);
	}
	a.at<double>(0, 0) = x2;
	a.at<double>(1, 0) = xiyi;
	a.at<double>(2, 0) = xi;
	a.at<double>(0, 1) = xiyi;
	a.at<double>(1, 1) = y2;
	a.at<double>(2, 1) = yi;
	a.at<double>(0, 2) = xi;
	a.at<double>(1, 2) = yi;
	a.at<double>(2, 2) = points3ds_.size();
	b.at<double>(0, 0) = zixi;
	b.at<double>(1, 0) = ziyi;
	b.at<double>(2, 0) = zi;
	//计算平面系数
	cv::Mat x = a.inv() * b;
	//Ax+by+cz=D
	this->C = 1;
	this->A = x.at<double>(0, 0);
	this->B = x.at<double>(1, 0);
	this->D = x.at<double>(2, 0);
	cout <<"  A=" << A << "  B=" << B << "  C=" << C << "  D=" << D << endl;
}

void LaserPlane::loadBoard(const vector<cv::Mat>& boardNoLaser, CalibrateCamera& camera) {
	rvecsMat_.resize(boardNoLaser.size());
	tvecsMat_.resize(boardNoLaser.size());
	std::vector<cv::Point2f> corner;
	//读取每幅图片的角点信息
	for (const auto& image : boardNoLaser) {
		findChessboardCorners(image, patternSize_, corner);
		cv::Mat gray = image.clone();
		if (image.channels() == 3)
			cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		//find4QuadCornerSubpix(gray, corner, cv::Size(5, 5));
		//drawChessboardCorners(image, patternSize_, corner, true);
		//imshow("BaseCalculate", image);
		//cv::waitKey();
		imagePoint_.push_back(corner);
		corner.clear();
	}
	std::vector<std::vector<cv::Point3f>> objRealPoint;
	camera.calRealPoint(objRealPoint,boardNoLaser.size(), 5);
	//计算两个视图的外参数 旋转向量和平移向量
	for (size_t n = 0; n < boardNoLaser.size(); n++) {
		//std::cout << "计算第" << n << "幅图片" << std::endl;
		solvePnP(
			objRealPoint.at(n),
			imagePoint_[n],
			camera.getCameraMatrix(),
			camera.getDistCoeff(),
			rvecsMat_[n],
			tvecsMat_[n],
			false,
			cv::SOLVEPNP_DLS
		);
	}
}


void LaserPlane::loadLaser(vector<cv::Mat>& boardLaser, CalibrateCamera& camera) {
	//对应的旋转矩阵
	auto it1 = rvecsMat_.begin();
	//对应的位移矩阵
	auto it2 = tvecsMat_.begin();
	//基准
	cv::Mat rotationMatrix2; //3*3
	Rodrigues(camera.getBaseRvecsMat(), rotationMatrix2);
	for (auto it0 = boardLaser.begin(); it0 != boardLaser.end(); ++it0, ++it1, ++it2) {
		std::vector<cv::Point3f> points3d;
		cv::Point3f temp;
		cv::Mat rodLightLine = *it0;
		//这里可以改成步长和种子点搜索法
		medianBlur(rodLightLine, rodLightLine, 5);
		cvtColor(rodLightLine, rodLightLine, cv::COLOR_BGR2GRAY);
		threshold(rodLightLine, rodLightLine, 25, 255, cv::THRESH_BINARY);
		//threshold(rodLightLine, rodLightLine, 0, 255, cv::THRESH_OTSU);
		//imshow("BaseCalculate", rodLightLine);
		//cv::waitKey();
		for (int i = 0; i < rodLightLine.rows; i++) {
			double sum = 0;
			int num = 0;
			for (int j = 0; j < rodLightLine.cols; j++)
				if (static_cast<int>(rodLightLine.at<uchar>(i, j)) == 255) {
					sum += j;
					num++;
				}
			if (num == 0)
				continue;
			//一张图片对应一个旋转向量和一个位移向量 
			//rvecsMat和tvecsMat对应22张图片的旋转向量和位移向量
			//将像素坐标转换为世界坐标，因为激光在标定板上，所以世界坐标的Z可以设为0，这个在计算s的时候会用到
			//输入了像素坐标
			//auto temp = cv::Point2f(i, 1.0 * sum / num);
			//points3ds_.push_back(camera.getCameraPoints(temp));

			temp = camera.getWorldPoints(cv::Point2f(sum / num, i), *it1, *it2);
			//到这里计算出的是激光条纹在标定板上的世界坐标，Z为0
			cv::Mat point3dMat = (cv::Mat_<double>(3, 1) << static_cast<double>(temp.x),
				static_cast<double>(temp.y), static_cast<double>(temp.z));
			//要转换的
			cv::Mat rotationMatrix1; //3*3
			Rodrigues(*it1, rotationMatrix1);
			//将两个不同的世界坐标转换到同一个基准下
			//s* Pc = M(RPo + T)
			//Pc=R(Po-T)
			//rotationMatrix1 * (Point3d_mat - *it2)计算出了世界坐标对应的相机坐标
			//Po=R-1*Pc+T
			//计算出了在基准下的世界坐标
			cv::Mat point3dToBaseMat = rotationMatrix2.inv() * rotationMatrix1 * (point3dMat - *it2)
				+ camera.getBaseTvecsMat();
			//
			//cout << "基准Z：" << point3dToBaseMat.at<double>(2, 0) << endl;
			points3ds_.emplace_back(
				point3dToBaseMat.at<double>(0, 0),
				point3dToBaseMat.at<double>(1, 0),
				point3dToBaseMat.at<double>(2, 0)
			);
		}
	}
}

vector<cv::Mat> LaserPlane::getRVecsMat() { return rvecsMat_; }

vector<cv::Mat> LaserPlane::getTVecsMat() { return tvecsMat_; }

double LaserPlane::getA() const { return A; }

double LaserPlane::getB() const { return B; }

double LaserPlane::getC() const { return C; }

double LaserPlane::getD() const { return D; }
