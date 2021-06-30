#ifndef SFMFUNCTIONS_H
#define SFMFUNCTIONS_H

#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

#include"matcher.h"
#include "vector"


using namespace std;
using namespace cv;

struct featurepoints{
    std::vector< cv::Point2f> point1,point2;
    Mat P;
};

class sfmfunctions
{
public:
    sfmfunctions();

    static Mat getCorrectCameraMatrix(Mat X, Mat F);
    static featurepoints findcorrectP(Mat img1, Mat img2, RobustMatcher rmatcher);

};

#endif // SFMFUNCTIONS_H
