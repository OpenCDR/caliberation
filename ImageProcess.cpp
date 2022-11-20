#include "ImageProcess.h"


std::vector<cv::Point2d> ProcessTool::averageLine(cv::Mat img0, const cv::Point2d& leftUp, const cv::Point2f& rightDown) const {
	const cv::Mat img = img0.clone();
	medianBlur(img0, img0, 5);
	cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
	threshold(img0, img0, 0, 255, cv::THRESH_OTSU);
	std::vector<cv::Point2d> result;
	for (int i = leftUp.x; i < rightDown.x; i++) {
		int sum = 0;
		int num = 0;
		for (int j = leftUp.y; j < rightDown.y; j++)
			if (img0.at<uchar>(j, i) == 255) {
				sum += j;
				num++;
			}
		if (num == 0)
			continue;
		result.emplace_back(i, 1.0 * sum / num);
	}
	showLine(result, img);
	return result;
}

std::vector<cv::Point2d> ProcessTool::stegerLine(cv::Mat img0, int col, int row, int sqrtX, int sqrtY, int threshold,
	float distance, bool isFloat) const {
	if (img0.channels() == 3)
		cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
	cv::Mat img;
	img = img0.clone();
	//��˹�˲�
	img.convertTo(img, CV_32FC1);
	//����,ѡ���߿�
	GaussianBlur(img, img, cv::Size(col, row), sqrtX, sqrtY);
	//cv::threshold(img, img, 70, 255, cv::THRESH_BINARY);
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
	//cv::morphologyEx(img, img, CV_MOP_OPEN, kernel);
	//��˹������õ�һ��ɽ��ǿ�ȵĹ���ͼ
	//��˹�����ʵ���Ͼ���һ����̬����
	// 
	//�����������ľ����
	//����ʹ�õ�����ֵ�����ƫ�������ǵõ��󵼹�ʽ
	cv::Mat m1, m2;
	m1 = (cv::Mat_<float>(1, 2) << 1, -1); //xƫ��������һ��Mat�����һ��ƫ��=f(x+1,y)-f(x,y)
	m2 = (cv::Mat_<float>(2, 1) << 1, -1); //yƫ�����������һ��ƫ��=f(x,y+1)-f(x,y)
	cv::Mat dx, dy;
	//ʹ��m1��� ��ǰ����ĻҶ�ֵ-��x+1���ĻҶ�ֵ
	filter2D(img, dx, CV_32FC1, m1); //���
	filter2D(img, dy, CV_32FC1, m2); //���
	//����ƫ����
	cv::Mat m3, m4, m5;
	m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1); //�����xƫ���ľ���=f(x+1,y)+f(x-1,y)-2f(x,y)
	m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1); //����yƫ���ľ���
	m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1); //����xyƫ������=f(x+1,y+1)-f(x+1,y)-f(x,y+1)+f(x,y)

	//Hessian������һ���Գƾ��󣬶������;���Ҳ�ǶԳƾ���
	cv::Mat dxx, dyy, dxy;
	//��ö�Ӧ���ƫ����ֵ
	filter2D(img, dxx, CV_32FC1, m3);
	filter2D(img, dyy, CV_32FC1, m4);
	filter2D(img, dxy, CV_32FC1, m5);

	//hessian����
	int imgCol = img.cols;
	int imgRow = img.rows;

	std::vector<cv::Point2d> pt;

	//
	for (int j = imgRow - 1; j >= 0; j--)
		for (int i = 0; i < imgCol; i++)
			//����һ����ֵ��ʱ��ʼ����
			if (img0.at<uchar>(cv::Point(i, j)) > threshold) {
				cv::Mat hessian(2, 2, CV_32FC1);
				//�õ���Ӧ���hessian����
				//��ÿ���㣬�����hessian����
				hessian.at<float>(0, 0) = dxx.at<float>(j, i);
				hessian.at<float>(0, 1) = dxy.at<float>(j, i);
				hessian.at<float>(1, 0) = dxy.at<float>(j, i);
				hessian.at<float>(1, 1) = dyy.at<float>(j, i);

				cv::Mat eValue;
				cv::Mat eVectors;
				//hessian����2*2 �Գƾ��� ����������ֵ
				//��ȡ��ǰ�������ֵ����������
				// ������������ֵ����Ӧ��������
				// ����ֵһ��Ϊ��һ��Ϊ����
				//		eValue: [3.3130465;
				//				-37.06522]
				//	eVectors : [0.89511645, 0.44583231;
				//				-0.44583231, 0.89511645]
				eigen(hessian, eValue, eVectors);
				//std::cout << "eValue:" << eValue << std::endl << "eVectors:" << eVectors << std::endl;
				//nxΪ����ֵ����Ӧ��x��nyΪ����ֵ����Ӧ��y
				double nx, ny;
				double maxD = 0;
				//������ֵ����ֵ���ʱ��Ӧ����������
				//Hessian���������ֵ�����������ڸõ㸽��������������İ�͹�ԣ�����ֵԽ��͹��Խǿ��
				if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0))) {
					nx = eVectors.at<float>(0, 0);
					ny = eVectors.at<float>(0, 1);
					maxD = eValue.at<float>(0, 0);
				}
				else {
					nx = eVectors.at<float>(1, 0);
					ny = eVectors.at<float>(1, 1);
					maxD = eValue.at<float>(1, 0);
				}
				// ʹ��̩��չ����ʾ��ǿ�ķֲ�����f(x+t*nx,y+t*ny)���Ա�ʾ��ǰ���긽���Ĺ�ǿ�ֲ�
				// f(x+t*nx,y+t*ny) hessian��������ֵ���Ķ�Ӧ����������(nx��ny)��ʾ�ݶ��������ķ��򣨹�ǿ�ı���죩��Ҫ���������ı�  
				// ����x��y��nx,ny��֪�������ǰ���긽���Ĺ�ǿ�ֲ��������Կ����ǹ���t��һ���ֲ�����
				// ����Զ�t�󵼣����Ϊ0�ĵط����ǹ�ǿ��ǿ�ĵط�����ֵ����Ҳ���Ǽ������ĵ㡣
				// ���t*nx��t*ny��С��0.5˵�������ֵ���λ�ڵ�ǰ�����ڣ�������ǿ�ֲ���������
				// ���t*nx��t*ny���ܴ󣬸�����ǿ�ֲ����������ã�����������㣬����ɨ����뼤�����ĸ����ĵ�
				double t = -
					(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i))
					/
					(nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));
				//t* nx��t* nx����������ĵ���뵱ǰ���صľ���
				if (fabs(t * nx) <= distance && fabs(t * ny) <= distance) {
					if (isFloat)
						pt.emplace_back(i + t * nx, j + t * ny);
					else
						pt.emplace_back(i, j);
				}
			}
	cvtColor(img0, img0, cv::COLOR_GRAY2BGR);
	showLine(pt, img0);
	return pt;
}

void ProcessTool::showLine(const std::vector<cv::Point2d>& points, cv::Mat image) const {
	for (const auto& point : points)
		image.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 0, 255);
	imshow("GetLine", image);
	cv::waitKey(1);
}
