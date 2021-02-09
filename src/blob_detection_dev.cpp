/*
 * A advanced example to detect blobs and separate close blobs by using cv::watershed
 * 
 * Author: Wen-Yu Chien, leochien1110@gmail.com
 * 
 * Reference: https://learnopencv.com/blob-detection-using-opencv-python-c/
 * blob split: https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
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
    imshow("Raw image",im);

    // enhance contrast
    threshold(im,im,0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //imshow("Contrast image",im);

    Mat im_inv;
    // binary inverse
    threshold(im,im_inv,0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    //imshow("Inverted image",im_inv);

    // distance transform
    Mat dist;
    distanceTransform(im,dist, DIST_L2,3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    //imshow("Distance Transform Image", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    //imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    circle(markers, Point(5,5), 3, Scalar(255), -1);
    //imshow("Markers", markers*10000);

    // Perform the watershed algorithm
    cvtColor(im,im,COLOR_GRAY2BGR);
    watershed(im, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    cvtColor(dst,dst,COLOR_BGR2GRAY);
    threshold(dst,dst,0, 255, CV_THRESH_BINARY_INV);
    // Visualize the final image
    imshow("Final Result", dst);


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
	detector->detect( dst, keypoints);
    
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