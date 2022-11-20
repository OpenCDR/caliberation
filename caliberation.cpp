#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int main() {
	ifstream inImgPath("calibdata.txt"); //标定所用图像文件的路径
	vector<string> imgList;
	vector<string>::iterator p;
	string temp;
	if (!inImgPath.is_open())
		cout << "没有找到文件" << endl;
	//读取文件中保存的图片文件路径，并存放在数组中
	while (getline(inImgPath, temp))
		imgList.push_back(temp);

	ofstream fout("caliberation_result.txt"); //保存标定结果的文件

	cout << "开始提取角点......" << endl;
	Size imageSize; //保存图片大小
	auto patternSize = Size(11, 8); //标定板上每行、每列的角点数；测试图片中的标定板上内角点数为8*6
	vector<Point2f> cornerPointsBuf; //建一个数组缓存检测到的角点，通常采用Point2f形式,以图像的左上角为原点，而不是以棋盘的左上角为原点
	vector<Point2f>::iterator cornerPointsBufPtr;
	vector<vector<Point2f>> cornerPointsOfAllImages; //所有图片的角点信息
	int imageNum = 0;
	string filename;
	while (imageNum < imgList.size()) {
		filename = imgList[imageNum++];
		cout << "image_num = " << imageNum << endl;
		cout << filename << endl;
		Mat imageInput = imread(filename);
		if (imageNum == 1) {
			imageSize.width = imageInput.cols;
			imageSize.height = imageInput.rows;
			cout << "image_size.width = " << imageSize.width << endl;
			cout << "image_size.height = " << imageSize.height << endl;
		}

		if (findChessboardCorners(imageInput, patternSize, cornerPointsBuf) == 0) {
			cout << "Can not find all chessboard corners!\n"; //找不到角点
			exit(1);
		}
		Mat gray;
		cvtColor(imageInput, gray, COLOR_RGB2GRAY); //将原来的图片转换为灰度图片
		find4QuadCornerSubpix(gray, cornerPointsBuf, Size(5, 5)); //提取亚像素角点,Size(5, 5),角点搜索窗口的大小。
		cornerPointsOfAllImages.push_back(cornerPointsBuf);
		drawChessboardCorners(gray, patternSize, cornerPointsBuf, true);
		namedWindow("camera calibration", 0);
		resizeWindow("camera calibration", 1080, 960);
		imshow("camera calibration", gray);
		waitKey(50);
	}

	int total = cornerPointsOfAllImages.size();
	cout << "total=" << total << endl;
	int cornerNum = patternSize.width * patternSize.height; //每张图片上的总的角点数
	// cout<<cornerNum<<endl;
	for (int i = 0; i < total; i++) {
		cout << "--> 第" << i + 1 << "幅图片的数据 -->:" << endl;
		for (int j = 0; j < cornerNum; j++) {
			cout << "-->" << cornerPointsOfAllImages[i][j].x;
			cout << "-->" << cornerPointsOfAllImages[i][j].y;
			if ((j + 1) % 3 == 0)
				cout << endl;
			else
				cout.width(10);
		}
		cout << endl;
	}

	cout << endl << "角点提取完成" << endl;
	destroyAllWindows();

	//摄像机标定
	cout << "开始标定………………" << endl;
	auto cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //内外参矩阵，H——单应性矩阵
	auto distCoefficients = Mat(1, 5, CV_32FC1, Scalar::all(0)); //摄像机的5个畸变系数：k1,k2,p1,p2,k3
	vector<Mat> tvecsMat; //每幅图像的平移向量，t，若干张图片的，不是一张图像的
	vector<Mat> rvecsMat; //每幅图像的旋转向量（罗德里格旋转向量，若干张图片的，不是一张图像的
	vector<vector<Point3f>> objectPoints; //保存所有图片的角点的三维坐标，所有图片的
	//初始化每一张图片中标定板上角点的三维坐标                 //世界坐标系，以棋盘格的左上角为坐标原点
	int j, k;
	//遍历每一张图片
	for (k = 0; k < imageNum; k++) {
		vector<Point3f> tempCornerPoints; //每一幅图片对应的角点数组
		//遍历所有的角点
		for (int i = 0; i < patternSize.height; i++)
			for (j = 0; j < patternSize.width; j++) {
				Point3f singleRealPoint; //一个角点的坐标，初始化三维坐标
				singleRealPoint.x = i * 10; //10是长/宽，根据黑白格子的长和宽，计算出世界坐标系（x,y,z)
				singleRealPoint.y = j * 10;
				singleRealPoint.z = 0; //假设z=0
				tempCornerPoints.push_back(singleRealPoint);
			}
		objectPoints.push_back(tempCornerPoints);
	}

	calibrateCamera(objectPoints, cornerPointsOfAllImages, imageSize, cameraMatrix, distCoefficients, rvecsMat,
	                tvecsMat, 0);
	cout << "标定完成" << endl;

	//开始保存标定结果
	cout << "开始保存标定结果" << endl;

	cout << endl << "相机相关参数：" << endl;
	fout << "相机相关参数：" << endl;
	cout << "1.内外参数矩阵:" << endl;
	fout << "1.内外参数矩阵:" << endl;
	cout << "大小：" << cameraMatrix.size() << endl;
	fout << "大小：" << cameraMatrix.size() << endl;
	cout << cameraMatrix << endl;
	fout << cameraMatrix << endl;
	//cout<<cameraMatrix.depth()<<endl;tempCornerPoints
	fout << "大小：" << distCoefficients.size() << endl;
	cout << distCoefficients << endl;
	fout << distCoefficients << endl;

	cout << endl << "图像相关参数：" << endl;
	fout << endl << "图像相关参数：" << endl;
	auto rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //旋转矩阵
	for (int i = 0; i < imageNum; i++) {
		cout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		cout << rvecsMat[i] << endl;
		fout << rvecsMat[i] << endl;
		cout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		Rodrigues(rvecsMat[i], rotationMatrix); //将旋转向量转换为相对应的旋转矩阵
		cout << rotationMatrix << endl;
		fout << rotationMatrix << endl;
		cout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		cout << tvecsMat[i] << endl;
		fout << tvecsMat[i] << endl;
	}

	cout << "结果保存完毕" << endl;

	//对标定结果进行评价
	cout << "开始评价标定结果......" << endl;

	//计算每幅图像中的角点数量，假设全部角点都检测到了
	int cornerPointsCounts;
	cornerPointsCounts = patternSize.width * patternSize.height;

	cout << "每幅图像的标定误差：" << endl;
	fout << "每幅图像的标定误差：" << endl;
	double err = 0; //单张图像的误差
	double totalErr = 0; //所有图像的平均误差
	for (int i = 0; i < imageNum; i++) {
		vector<Point2f> imagePointsCalculated; //存放新计算出的投影点的坐标
		vector<Point3f> tempPointSet = objectPoints[i];
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoefficients, imagePointsCalculated);
		//计算根据内外参等重投影出来的新的二维坐标，输出到image_points_calculated
		//计算新的投影点与旧的投影点之间的误差
		vector<Point2f> imagePointsOld = cornerPointsOfAllImages[i]; //向量
		//将两组数据换成Mat格式
		auto imagePointsCalculatedMat = Mat(1, imagePointsCalculated.size(), CV_32FC2); //将mat矩阵转成1维的向量
		auto imagePointsOldMat = Mat(1, imagePointsOld.size(), CV_32FC2);
		for (j = 0; j < tempPointSet.size(); j++) {
			imagePointsCalculatedMat.at<Vec2f>(0, j) = Vec2f(imagePointsCalculated[j].x,
			                                                    imagePointsCalculated[j].y); //vec2f->一个二维的向量
			imagePointsOldMat.at<Vec2f>(0, j) = Vec2f(imagePointsOld[j].x, imagePointsOld[j].y); //直接调用函数，不用定义对象
		}
		err = cv::norm(imagePointsCalculatedMat, imagePointsOldMat, NORM_L2); //输入的是矩阵
		err /= cornerPointsCounts; //每个角点的误差
		totalErr += err;
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << totalErr / imageNum << "像素" << endl;
	fout << "总体平均误差：" << totalErr / imageNum << "像素" << endl;
	cout << "评价完成" << endl;


	//下面就是矫正图像的代码
	auto mapX = Mat(imageSize, CV_32FC1); //对应坐标的重映射参数
	auto mapY = Mat(imageSize, CV_32FC1);
	Mat r = Mat::eye(3, 3, CV_32F); //定义的对角线为1的对角矩阵
	cout << "保存矫正图像" << endl;
	string imageFileName;
	std::stringstream strStm;
	for (int i = 0; i < imageNum; i++) {
		cout << "Frame #" << i + 1 << endl;
		initUndistortRectifyMap(cameraMatrix, distCoefficients, r, cameraMatrix, imageSize, CV_32FC1, mapX, mapY);
		//输入内参，纠正后的内参，外参等,计算输出矫正的重映射参数(相片尺寸 width*height),每个像素点都需要转换
		Mat srcImage = imread(imgList[i], 1);
		Mat newImage = srcImage.clone();
		remap(srcImage, newImage, mapX, mapY, INTER_LINEAR);
		namedWindow("原始图像", 0);
		resizeWindow("原始图像", 1080, 960);
		namedWindow("矫正后图像", 0);
		resizeWindow("矫正后图像", 1080, 960);
		imshow("原始图像", srcImage);
		imshow("矫正后图像", newImage);

		strStm.clear();
		imageFileName.clear();
		strStm << i + 1;
		strStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite(imageFileName, newImage);

		waitKey(200);
	}
	destroyAllWindows();
	cout << "保存结束" << endl;
	fout.close(); //
	waitKey(0);

	return 0;
}
