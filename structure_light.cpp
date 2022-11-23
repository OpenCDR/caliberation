// use for calibration
#include "structure_light.h"

#include <utility>
#include "struct_light_calib.h"

structure_light::structure_light(const int x, cv::Size patternSize, cv::Size patternLength) {
	imageCount = x;
	structure_light::patternSize = std::move(patternSize);
	PatternLength = std::move(patternLength);
}

void structure_light::generateCalibBoardPoint() {
	for (int i = 0; i < imageCount; i++)
		if (isRecognize[i]) {
			std::vector<cv::Point3f> tempPoint;
			for (int j = 0; j < patternSize.height; j++)
				for (int k = 0; k < patternSize.width; k++) {
					cv::Point3f temp;
					temp.x = k * PatternLength.width;
					temp.y = j * PatternLength.height;
					temp.z = 0;
					tempPoint.push_back(temp);
				}
			calibBoardPoint.push_back(tempPoint);
			tempPoint.clear();
		}
}

void structure_light::outputResult() const {
	cv::FileStorage out("Parameters.xml", cv::FileStorage::WRITE);
	out << "cameraMatrix" << cameraMatrix;
	out << "distCoeffs" << distCoeffs;
	out << "Formular_A" << planeFormular[0];
	out << "Formular_B" << planeFormular[1];
	out << "Formular_C" << planeFormular[2];
	out << "Formular_D" << planeFormular[3];
	out << "Rw" << Rw;
	out << "Tw" << Tw;
}
