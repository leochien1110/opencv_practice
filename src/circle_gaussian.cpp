/*
 * A simple example to demo the circle and gaussian blur
 * 
 * Author: Wen-Yu Chien, leochien1110@gmail.com
 */

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream> 

int main(){
    
    cv::namedWindow("Circles",CV_WINDOW_NORMAL);
    cv::Mat img(480,640,CV_8UC3, cv::Scalar(0,0,0));

    cv::circle(img,cv::Point(200,200),40,cv::Scalar(255,255,255),CV_FILLED,8,0);
    cv::boxFilter(img,img,-1, cv::Size(20,20));
    
    cv::circle(img,cv::Point(400,200),40,cv::Scalar(255,255,255),CV_FILLED,8,0);

    cv::imshow("Circles",img);
    cv::waitKey(0);

}