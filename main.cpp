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

//����һЩȫ�ֱ�����Ϊ�궨�İб����
int imageCount = 11; //ͼ�������
cv::Size patternSize(7, 7); //�б꣨Բ��������
cv::Size2f patternLength(37.5, 37.5); //����Բ�α��֮��ľ���
//patternType:
//0:circle;
//1:chessboard;
bool isCircle = true;

int main() {
	structure_light lineStructureLight(imageCount, patternSize, patternLength);

	double reProjectionError = 0.0;
	cameraCalib(lineStructureLight, reProjectionError); //������궨
	cout << "Camera calibration reProjection error: " << reProjectionError << " pixels." << endl;

	//����������ȡ
	//steger����������ȡ�㷨
	stegerLine(lineStructureLight);

	//�ṹ�����ֱ��������ཻ��������ȡ
	vector<vector<cv::Point2f>> intersectPoint; //��ƽ��Ͱб�ƽ���ϵĵ�
	crossPoint(lineStructureLight, intersectPoint);
	//���Ȳ�����ȡ��ƽ����ά����
	crossRatio(lineStructureLight, intersectPoint);

	//��Ϲ�ƽ��
	lightPlainFitting(lineStructureLight);

	// ������Ƶ�����
	cout << lineStructureLight.lightPlanePoint.size() << endl;

	// �����ƴ洢��һ��TXT�ļ���
	ofstream outfile;
	outfile.open("pointCloud.txt", ios::binary | ios::app | ios::in | ios::out);
	for (auto& k : lineStructureLight.lightPlanePoint) {
		outfile << k.x << " ";
		outfile << k.y << " ";
		outfile << k.z << "\n";
	}
	outfile.close(); //�ر��ļ��������ļ�
	//����궨���
	lineStructureLight.outputResult();

	//�Ա궨��������������������෴ͶӰ��
	Structlight structLightMeasure;
	structLightMeasure.readParameters();
	vector<vector<float>> backProjectError;
	estimateError2(lineStructureLight, structLightMeasure, backProjectError);
	cout << "finish!" << endl;
	return 0;

}
