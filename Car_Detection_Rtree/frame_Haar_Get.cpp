#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/ml/ml.hpp>
#include <iostream>  
#include "Haar.h" 
#include <string>  
#include <vector>  
#include "omp.h"  


using namespace std;
using namespace cv;


extern vector<double> window_position;				// Œª÷√±Í«©£¨¥Ê¥¢À≥–Ú£∫x, y, width, height
extern vector<Rect> windowrect;					// Œª÷√±Í«©


extern float Getting_Haar_From_frame(Mat& _frame, vector<Point>& _car, CvRTrees& rtree, const int& feat_n = 3115, Point& f_postion = Point(0, 0), const int& win_width = 200, const int& win_height = 200,
									const int& step = 20, const int& win_size = 24)
{
	HaarFeature m_feature(1, 6);
	//Rect dwindow = Rect(0, 0, win_width, win_height);
	Mat test_data = Mat(1, feat_n, CV_32FC1);
	float num = 0;
	float car_n = 0;
	for (float x = 0; x <= _frame.cols - win_width; x = x + step) 
	{
		for (float y = 0; y <= _frame.rows - win_height; y = y + step)
		{
			Rect dwindow = Rect(x, y, win_width, win_height);
			Mat frame_detect = _frame(dwindow);
			Point pos = Point(x, y);
			resize(frame_detect, frame_detect, Size(win_size, win_size));
			num++;

			Rect rect = Rect(0, 0, 0, 0);
			vector<float> m_feat;
			m_feature.caluHf(frame_detect, m_feat);






					//m_feat = m_feature[3].caluHf(frame_detect);

					for (int i = 0; i < m_feat.size(); i++)
					{
						//cout << i << "|";
						//cout << m_feat << endl;
						if (m_feat[i] < -1.70141e+38 || m_feat[i] > 1.70141e+38)
						{
							cout << i << endl;
						}
						test_data.at<float>(0, i) = m_feat[i];
					}


					float result = rtree.predict(test_data, Mat());


					if (fabs(result - 1) <= FLT_EPSILON)
					{
						car_n++;
						_car.push_back(pos);
					}

			}
		}


		return car_n/num;
}