#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/ml/ml.hpp>
#include "haar.h"
#include <iostream>  
#include <string>  
#include <vector>  
#include <io.h>


using namespace std;
using namespace cv;

S_Haar Single_Haar;							// 单个窗口的 Haar 特征值存储
vector<S_Haar> All_Haar;					// 所有 Haar 特征值的存储
vector<double> window_position;				// 位置标签，存储顺序：x, y, width, height
vector<Rect> windowrect;					// 位置标签


void Getting_Haar_From_frame(Mat& _frame, vector<Point>& _car, Point& f_postion = Point(0, 0), const int& win_width = 200, const int& win_height = 200,
									const int& scale = 20, const int& feat_num = 6349);
int read_Haar_from_vector(Mat& _data, vector<S_Haar>& src, Mat& _class);



const char Rtree_name[] = "1.xml";
String filename = "x.avi";
CvRTrees* rtree = new CvRTrees;
string window_name = "Rtree_car_detection";


int main(int argc, char* argv[])
{
	Mat frame = imread("..\\input\\cars_sample001.bmp");
	Mat dframe = frame;
	rtree->clear();
	rtree->load("../use_xml/one.xml");
	//VideoCapture capture(filename);
	//capture >> frame;
	//int x_size = 128, y_size = 128;
	CvSize size = cvSize(128, 128);			// Cvsize r = cvSize(int width, int height)
	double time0 = static_cast<double>(getTickCount());
	Getting_Haar_From_frame(dframe, 1, size);
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "提取特征运行时间为：" << time0 << "秒" << endl;


	cout << "Done Getting Haar!" << endl;


	vector<Rect> car;
	Mat detect_data = Mat(All_Haar.size(), 5859, CV_32FC1);			// 特征集Mat
	Mat detect_class = Mat(All_Haar.size(), 1, CV_32FC1);		// 标签矩阵
	double result;
	if (read_Haar_from_vector(detect_data, All_Haar, detect_class))
	{
		Mat detect_sample;
		for (int dsample = 0; dsample < All_Haar.size(); dsample++)
		{
			detect_sample = detect_data.row(dsample);
			result = rtree->predict(detect_sample, Mat());
			//printf("Testing Sample %i -> class result (digit %d)\n", dsample, (int) result);
			// (N.B. openCV uses a floating point decision tree implementation!)  
			if (fabs(result - 1 )  <= FLT_EPSILON )
			{
				int x = window_position[4 * dsample];
				int y = window_position[4 * dsample + 1];
				int width = window_position[4 * dsample + 2];
				int height = window_position[4 * dsample + 3];
				// if they differ more than floating point error => wrong class  
				car.push_back(windowrect[dsample]);
				cout << dsample << ":" << x << "|" << y << "|" << width << "|" << height << endl;
			}
			else
			{

			}
		}
	}

	for (size_t i = 0; i<car.size(); i++)
	{
		//if ((car[i].height > 80) && (car[i].width > 80))
		{
			/*	cout << car[i].height << " " << car[i].width << endl;*/
			rectangle(frame, car[i], Scalar(255, 0, 255), 2);
		}
		//else
		//{
		//	continue;
		//}
	}
	imshow(window_name, frame);
	waitKey(0);

	return 0;
}


int read_Haar_from_vector(Mat& _data, vector<S_Haar>& src, Mat& _class)
{
	S_Haar tmp;
	for (int num = 0; num < src.size(); num++)
	{
		tmp = src[num];
		for (int _row = 0; _row < src[num].size(); _row++)
		{
			if (_row < src[num].size() - 1)
			{
				_data.at<float>(num, _row) = tmp[_row];
			}
			else if (_row = src[num].size() - 1)
			{
				_class.at<float>(num, 0) = tmp[_row];
			}
		}
	}
	return 1;
}


