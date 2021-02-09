/*
 * A simple example to detect blobs 
 * 
 * Author: Wen-Yu Chien, leochien1110@gmail.com
 * 
 * Reference: https://learnopencv.com/blob-detection-using-opencv-python-c/
 */

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    
    // Read Image
    Mat im = imread("../res/beacons.png", IMREAD_GRAYSCALE);
    if(im.empty()){
        cout << "Could not load the image!\n" << endl;
        return -1;
    }
    //imshow("Raw image",im);

    // enhance contrast
    threshold(im,im,0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Contrast image",im);

    Mat im_inv;
    // binary inverse, blob detection can only detect black blobs
    threshold(im,im_inv,0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    imshow("Inverted image",im_inv);

    // Setup Simple blob detector parameters
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold =  10;  //black
    params.maxThreshold = 200;  //white

    // Filer by Area
    params.filterByArea = true;
    params.minArea = 500;
    params.maxArea = 500000;

    // Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.83;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;


	// Storage for blobs
	vector<KeyPoint> keypoints;

    // Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

    // Detect blobs
	detector->detect( im_inv, keypoints);
    
    // print keypoints
    cout << "size of keypoints:" << keypoints.size() << endl;

    for(int i =0; i < keypoints.size(); i++){
        cout << "keypoints:" << keypoints[i].pt.x << "," << keypoints[i].pt.y << endl;
    }

    // Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// Show blobs
	imshow("keypoints", im_with_keypoints );
	waitKey(0);

}