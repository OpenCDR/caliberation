#ifndef STRUCT_LIGHT_CALIB_H
#define STRUCT_LIGHT_CALIB_H


#include <vector>
#include <opencv2/core/types.hpp>

#include "structlight.h"

//#include "structure_light.h"

using namespace std;

class structure_light;

int cameraCalib(structure_light& a, double& reProjectionError);

void stegerLine(structure_light& a);
//void greyCenter(Mat &image, structure_light &a);

void crossPoint(const structure_light& a, std::vector<std::vector<cv::Point2f>>& crossPoint);

void crossRatio(structure_light& light, std::vector<std::vector<cv::Point2f>>& crossPoint);

void lightPlainFitting(structure_light& light);

void extratLightLine(int imageIndex, cv::Mat& outputImage, structure_light& a);

void calc3DCoordination(Structlight a, const vector<cv::Point2f>& centerPoint, vector<cv::Point3f>& calcPoint);

void estimateError(const structure_light& structureLightCalib, const Structlight& structLightMeasure, vector<vector<float>> backProjectError);

void estimateError2(structure_light structureLightCalib, const Structlight& structLightMeasure, vector<vector<float>> backProjectError);

double get_norm(double* x, int n);
double normalize(double* x, int n);
inline double product(double* a, double* b, int n);
void orth(double* a, double* b, int n);
bool svd(vector<vector<double> > A, int K, vector<vector<double> >& U, vector<double>& S, vector<vector<double> >& V);

#endif //STRUCT_LIGHT_CALIB_H
