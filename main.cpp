#include <QCoreApplication>
#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "vector"


#include "matcher.h"
#include "sfmfunctions.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    IplImage *ipl;
    CvCapture* capture = cvCreateCameraCapture(0);
    Mat frame,matref,matreceived, F1,P;
    int ncapture = 0;

    featurepoints fp;



    RobustMatcher rmatcher;
    rmatcher.setConfidenceLevel(0.98);
    rmatcher.setMinDistanceToEpipolar(1.0);
    rmatcher.setRatio(0.65f);
    cv::Ptr<cv::FeatureDetector> pfd= new cv::SurfFeatureDetector(10);
    rmatcher.setFeatureDetector(pfd);

    while(capture)
    {
      ipl = cvQueryFrame(capture);
      frame = cvarrToMat(ipl);
      imshow("Video Frame", frame);

      char ch =  cvWaitKey(25);

      if(ch == 's')
      {
          ncapture++;


          if(ncapture == 1)
          {
              cout << "Capturing frame "<< ncapture << endl;
              matref = frame.clone();
              imshow("Capture 1", matref);

          }
          else if(ncapture == 2)
          {
              Mat P1 = (Mat_<double>(3,4)<< 1,0,0,0,0,1,0,0,0,0,1,0);
              cout << "Capturing frame "<< ncapture << endl;
              matreceived = frame.clone();
              imshow("Capture 2", matreceived);
              fp = sfmfunctions::findcorrectP(matref,matreceived,rmatcher);

              cout <<"P= "<< fp.P << endl;
              Mat Xw;

              triangulatePoints(P1,fp.P,fp.point1,fp.point2,Xw);

              cout << "Xw= " << Xw << endl;





          }
          else
          {
              cout << "Capturing frames: 2" << endl;
          }

      }
     if ( (cvWaitKey(10)) == 27 ) break;
    }



    return a.exec();
}
