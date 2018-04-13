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

S_Haar Single_Haar;							// �������ڵ� Haar ����ֵ�洢
vector<S_Haar> All_Haar;					// ���� Haar ����ֵ�Ĵ洢
vector<double> window_position;				// λ�ñ�ǩ���洢˳��x, y, width, height
vector<Rect> windowrect;					// λ�ñ�ǩ


float Getting_Haar_From_frame(Mat& _frame, vector<Point>& _car, CvRTrees& rtree, const int& feat_n = 3115, Point& f_postion = Point(0, 0),
	const int& win_width = 200, const int& win_height = 200, const int& step = 20, const int& win_size = 24);
int read_Haar_from_vector(Mat& _data, vector<S_Haar>& src, Mat& _class);



const char Rtree_name[] = "1.xml";
String filename = "x.avi";
CvRTrees rtree;
string window_name = "Rtree_car_detection";


int main(int argc, char* argv[])
{
	Mat frame = imread("..\\input\\sign01.bmp");
	Mat dframe = frame;

	//Mat imageRGB[3];
	//split(dframe, imageRGB);
	//for (int i = 0; i < 3; i++)
	//{
	//	equalizeHist(imageRGB[i], imageRGB[i]);
	//}
	//merge(imageRGB, 3, dframe);

	cvtColor(dframe, dframe, CV_RGB2GRAY);


	rtree.clear();
	rtree.load("../use_xml/4-1.xml");
	Rect split0 = Rect(0, 0, dframe.cols, dframe.rows / 2);
	Rect split1 = Rect(0, dframe.rows / 2, dframe.cols, dframe.rows / 2);
	Mat frame0 = dframe(split0);
	Mat frame1 = dframe(split1);
	
	
	vector<Point> car0;
	vector<Point> car1;
	double time0 = static_cast<double>(getTickCount());
	float a;
#pragma omp parallel sections  
{
#pragma omp section
	{
		Getting_Haar_From_frame(frame0, car0, rtree,13711); 
	}
#pragma omp section
	{
		a = Getting_Haar_From_frame(frame1, car1, rtree,13711);
	}
}
cout << a << endl;
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "����ʱ��Ϊ��" << time0 << "��" << endl;


	for (size_t i = 0; i<car1.size(); i++)
	{
		Rect _rect = Rect(car1[i].x, car1[i].y + dframe.rows/2, 200, 200);
		//if ((car[i].height > 80) && (car[i].width > 80))
		{
			/*	cout << car[i].height << " " << car[i].width << endl;*/
			rectangle(frame, _rect, Scalar(255, 0, 255), 2);
		}
		//else
		//{
		//	continue;
		//}
	}

	for (size_t i = 0; i<car0.size(); i++)
	{
		Rect _rect = Rect(car0[i].x, car0[i].y, 200, 200);
		//if ((car[i].height > 80) && (car[i].width > 80))
		{
			/*	cout << car[i].height << " " << car[i].width << endl;*/
			rectangle(frame, _rect, Scalar(255, 0, 255), 2);
		}
		//else
		//{
		//	continue;
		//}
	}

	//VideoCapture capture(filename);
	//capture >> frame;
	imshow("1", frame);
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


