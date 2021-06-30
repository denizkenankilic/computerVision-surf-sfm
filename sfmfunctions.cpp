#include "sfmfunctions.h"
#include "iostream"
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
#include "sfmfunctions.h"


using namespace std;
using namespace cv;
sfmfunctions::sfmfunctions()
{
    std::cout << "Welcom to SFM "<< std::endl;
}
double opensign(double sign)
{

            if( sign < 0 )
                sign = -1;

            else if (sign > 0 )
                sign = 1;

            else if(sign == 0)
                sign = 0;


    return sign;
}

Mat sfmfunctions::getCorrectCameraMatrix(Mat X, Mat F)
{
    Mat K1 = (Mat_<double>(3,3)<< 532.574, 0,       318.264,
                                             0,     531.361 ,228.96,
                                             0,     0,       1);

    Mat W = (Mat_<double>(3,3)<< 0,-1,0,1,0,0,0,0,1);

    Mat E = K1.t()*F*K1;

    // cout << "F=" << F << endl;

    Mat fundemantal = F.clone();

    SVD svd(E);
    //Find 4 possible P



    Mat R1 = svd.u*W*svd.vt;
    Mat R2 = svd.u*W.t()*svd.vt;
    Mat T1 = svd.u.col(2);
    Mat T2 = -svd.u.col(2);



    Mat P43 = (Mat_<double>(3,4)<<R1.at<double>(0,0),R1.at<double>(0,1),R1.at<double>(0,2),T1.at<double>(0,0),
                                  R1.at<double>(1,0),R1.at<double>(1,1),R1.at<double>(1,2),T1.at<double>(1,0),
                                  R1.at<double>(2,0),R1.at<double>(2,1),R1.at<double>(2,2),T1.at<double>(2,0));

    Mat P44 = (Mat_<double>(3,4)<<R1.at<double>(0,0),R1.at<double>(0,1),R1.at<double>(0,2),T2.at<double>(0,0),
                                  R1.at<double>(1,0),R1.at<double>(1,1),R1.at<double>(1,2),T2.at<double>(1,0),
                                  R1.at<double>(2,0),R1.at<double>(2,1),R1.at<double>(2,2),T2.at<double>(2,0));

    Mat P41 = (Mat_<double>(3,4)<<R2.at<double>(0,0),R2.at<double>(0,1),R2.at<double>(0,2),T1.at<double>(0,0),
                                  R2.at<double>(1,0),R2.at<double>(1,1),R2.at<double>(1,2),T1.at<double>(1,0),
                                  R2.at<double>(2,0),R2.at<double>(2,1),R2.at<double>(2,2),T1.at<double>(2,0));

    Mat P42 = (Mat_<double>(3,4)<<R2.at<double>(0,0),R2.at<double>(0,1),R2.at<double>(0,2),T2.at<double>(0,0),
                                  R2.at<double>(1,0),R2.at<double>(1,1),R2.at<double>(1,2),T2.at<double>(1,0),
                                  R2.at<double>(2,0),R2.at<double>(2,1),R2.at<double>(2,2),T2.at<double>(2,0));


//    cout << "P41= " << P41 << endl;
//    cout << "P42= " << P42 << endl;
//    cout << "P43= " << P43 << endl;
//    cout << "P44= " << P44 << endl;



    Mat x = X.col(0);
    Mat xp = X.col(1);

    Mat Pcam = (Mat_<double>(3,4) << 1,0,0,0,
                                     0,1,0,0,
                                     0,0,1,0);
    Mat P = K1*Pcam;

    Mat xhat = K1.inv()*x;

    Mat X3D = Mat::zeros(4,4,svd.vt.type());
    Mat Depth= (Mat_<double>(4,2)<< 0,0,
                                    0,0,
                                    0,0,
                                    0,0);
    Mat PXcam;

    for(int i = 0; i<4; i++)
    {
        if(i==0)
        {
            PXcam = P41.clone();
        }
        if (i == 1)
        {
            PXcam = P42.clone();}
        if (i == 2){
            PXcam = P43.clone();}
        if(i==3){
            PXcam = P44.clone();}
        //cout<<"lalalala"<<PXcam<<endl;

        Mat xphat = K1.inv()*xp;


        // We build the matrix A

        Mat A = (Mat_<double>(4,4)<<Pcam.at<double>(2,0)*xhat.at<double>(0,0)-Pcam.at<double>(0,0),Pcam.at<double>(2,1)*xhat.at<double>(0,0)-Pcam.at<double>(0,1),Pcam.at<double>(2,2)*xhat.at<double>(0,0)-Pcam.at<double>(0,2),Pcam.at<double>(2,3)*xhat.at<double>(0,0)-Pcam.at<double>(0,3),
                                    Pcam.at<double>(2,0)*xhat.at<double>(1,0)-Pcam.at<double>(1,0),Pcam.at<double>(2,1)*xhat.at<double>(1,0)-Pcam.at<double>(1,1),Pcam.at<double>(2,2)*xhat.at<double>(1,0)-Pcam.at<double>(1,2),Pcam.at<double>(2,3)*xhat.at<double>(1,0)-Pcam.at<double>(1,3),
                                    PXcam.at<double>(2,0)*xphat.at<double>(0,0)-PXcam.at<double>(0,0),PXcam.at<double>(2,1)*xphat.at<double>(0,0)-PXcam.at<double>(0,1),PXcam.at<double>(2,2)*xphat.at<double>(0,0)-PXcam.at<double>(0,2),PXcam.at<double>(2,3)*xphat.at<double>(0,0)-PXcam.at<double>(0,3),
                                    PXcam.at<double>(2,0)*xphat.at<double>(1,0)-PXcam.at<double>(1,0),PXcam.at<double>(2,1)*xphat.at<double>(1,0)-PXcam.at<double>(1,1),PXcam.at<double>(2,2)*xphat.at<double>(1,0)-PXcam.at<double>(1,2),PXcam.at<double>(2,3)*xphat.at<double>(1,0)-PXcam.at<double>(1,3));


//        cout << "A= "<< A << endl;
//        //Normalize A

        double A1n = sqrt(A.at<double>(0,0)*A.at<double>(0,0)+A.at<double>(0,1)*A.at<double>(0,1)+A.at<double>(0,2)*A.at<double>(0,2)+A.at<double>(0,3)*A.at<double>(0,3));
        double A2n = sqrt(A.at<double>(1,0)*A.at<double>(1,0)+A.at<double>(1,1)*A.at<double>(1,1)+A.at<double>(1,2)*A.at<double>(1,2)+A.at<double>(1,3)*A.at<double>(1,3));
        double A3n = sqrt(A.at<double>(0,0)*A.at<double>(0,0)+A.at<double>(0,1)*A.at<double>(0,1)+A.at<double>(0,2)*A.at<double>(0,2)+A.at<double>(0,3)*A.at<double>(0,3));
        double A4n = sqrt(A.at<double>(0,0)*A.at<double>(0,0)+A.at<double>(0,1)*A.at<double>(0,1)+A.at<double>(0,2)*A.at<double>(0,2)+A.at<double>(0,3)*A.at<double>(0,3));
//        cout << "A1norm= " << A1n << endl;
//        cout << "A2norm= " << A2n << endl;
//        cout << "A3norm= " << A3n << endl;
//        cout << "A4n= " << A4n << endl;
        Mat Anorm(4,4,A.type());

        Anorm.row(0) = A.row(0)/A1n;
        Anorm.row(1) = A.row(1)/A2n;
        Anorm.row(2) = A.row(2)/A3n;
        Anorm.row(3) = A.row(3)/A4n;

//        cout << "Anorm= " << Anorm << endl;






        SVD as(Anorm);
        //cout << "Vt= " << as.vt << endl;
        Mat asd = as.vt.t().col(as.vt.cols-1);

        X3D.at<double>(0,i) = asd.at<double>(0,0);
        X3D.at<double>(1,i) = asd.at<double>(1,0);
        X3D.at<double>(2,i) = asd.at<double>(2,0);
        X3D.at<double>(3,i) = asd.at<double>(3,0);

        Mat xi = PXcam* X3D.col(i);

        double w = xi.at<double>(2,0);

        double T = X3D.at<double>(X3D.rows-1,i);

        double m3n = sqrt(PXcam.at<double>(2,0)*PXcam.at<double>(2,0)+PXcam.at<double>(2,1)*PXcam.at<double>(2,1)+PXcam.at<double>(2,2)*PXcam.at<double>(2,2));

        Depth.at<double>(i,0) = (opensign(determinant(PXcam.colRange(0,3)))*w)/(T*m3n);

        xi = Pcam* X3D.col(i);

        w = xi.at<double>(2,0);

        T = X3D.at<double>(X3D.rows-1,i);

        m3n = sqrt(Pcam.at<double>(2,0)*Pcam.at<double>(2,0)+Pcam.at<double>(2,1)*Pcam.at<double>(2,1)+Pcam.at<double>(2,2)*Pcam.at<double>(2,2));

        Depth.at<double>(i,1) = (opensign(determinant(Pcam.colRange(0,3)))*w)/(T*m3n);



    }

     //cout << "Depth= " << Depth << endl;

    if(Depth.at<double>(0,0)>0 && Depth.at<double>(0,1)>0)
        P = P41.clone();
    else if (Depth.at<double>(1,0)>0 && Depth.at<double>(1,1)>0)
        P = P42.clone();
    else if (Depth.at<double>(2,0)>0 && Depth.at<double>(2,1)>0)
        P = P43.clone();
    else if (Depth.at<double>(3,0)>0 && Depth.at<double>(3,1)>0)
        P = P44.clone();
    else
        cout << "Not found!" << endl;


return P;
}
featurepoints sfmfunctions::findcorrectP(Mat img1, Mat img2, RobustMatcher rmatcher)
{
    Mat P;
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat fundemental= rmatcher.match(img1,img2,matches, keypoints1, keypoints2);


    // draw the matches
    cv::Mat imageMatches;
    cv::drawMatches(img1,keypoints1,  // 1st image and its keypoints
                        img2,keypoints2,  // 2nd image and its keypoints
                                    matches,                        // the matches
                                    imageMatches,           // the image produced
                                    cv::Scalar(255,255,255)); // color of the lines
    cv::namedWindow("Matches");
    cv::imshow("Matches",imageMatches);


    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
             it!= matches.end(); ++it) {

                     // Get the position of left keypoints
                     float x= keypoints1[it->queryIdx].pt.x;
                     float y= keypoints1[it->queryIdx].pt.y;
                     points1.push_back(cv::Point2f(x,y));
                     // Get the position of right keypoints
                     x= keypoints2[it->trainIdx].pt.x;
                     y= keypoints2[it->trainIdx].pt.y;
                     points2.push_back(cv::Point2f(x,y));



    }
    double a1 = points1[10].x;
    double a2 = points1[10].y;
    double a3 = points2[10].x;
    double a4 = points2[10].y;
    Mat inX;






    cv::Mat fundemantal;
    if(points2.size() > 30 && points1.size() > 30)
    {
       inX = (Mat_<double>(3,2) << a1,a3,a2,a4,1,1);
         cout << "inX= " << inX << endl;

    std::vector<uchar> inliers(points1.size(),0);
     fundemantal= cv::findFundamentalMat(
            cv::Mat(points2),cv::Mat(points1), // corresponding points
            inliers,        // outputed inliers matches
            CV_RANSAC,      // RANSAC method
            0.01,0.99);        // max distance to reprojection point

            P = sfmfunctions::getCorrectCameraMatrix(inX,fundemantal);
    }
    else
    {
        cv::Mat T = (cv::Mat_<double>(3, 3) <<
            1, 0, 0,
            0, 1, 0,
            0, 0, 1);
        std::cout<<"There is no P!"<<std::endl;

        fundemantal = T;
    }


    featurepoints fp;
    fp.P = P;
    fp.point1 = points1;
    fp.point2 = points2;






    return fp;
