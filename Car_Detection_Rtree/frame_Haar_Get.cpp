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


extern S_Haar Single_Haar;							// 单个窗口的 Haar 特征值存储,S_Haar的声明放于 Haar.h 头文件中
extern vector<S_Haar> All_Haar;					// 所有 Haar 特征值的存储
extern vector<double> window_position;				// 位置标签，存储顺序：x, y, width, height
extern vector<Rect> windowrect;					// 位置标签


extern void Getting_Haar_From_frame(Mat& _frame, vector<Point>& _car, CvRTrees& rtree,Point& f_postion = Point(0, 0), const int& win_width = 200, const int& win_height = 200,
									const int& scale = 20, const int& feat_num = 6349)
{
	//Rect dwindow = Rect(0, 0, win_width, win_height);
	Mat test_data = Mat(1, 6349, CV_32FC1);
	for (float x = 0; x <= _frame.cols - win_width; x = x + scale) 
	{
		for (float y = 0; y <= _frame.rows - win_height; y = y + scale)
		{
			Rect dwindow = Rect(x, y, win_width, win_height);
			Mat frame_detect = _frame(dwindow);
			Point pos = Point(x, y);



			Rect rect = Rect(0, 0, 0, 0);
			int size_max = 12;
			int x_Max, y_Max;
			vector<HaarFeature> m_feature;


					for (int size = 2; size <= size_max; size = size + 2)
					{
						//cout << endl << "尺度大小" << size << endl;
						for (int type = 0; type <= 7; type++)
						{
							switch (type)
							{
							case 0:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 2 * size;
								break;
							}
							case 1:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 2 * size;
								break;
							}
							case 2:
							{
								x_Max = frame_detect.cols - 3 * size;
								y_Max = frame_detect.rows - 2 * size;
								break;
							}
							case 3:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 3 * size;
								break;
							}
							case 4:
							{
								x_Max = frame_detect.cols - 4 * size;
								y_Max = frame_detect.rows - 2 * size;
								break;
							}
							case 5:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 4 * size;
								break;
							}
							case 6:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 4 * size;
								break;
							}
							case 7:
							{
								x_Max = frame_detect.cols - 2 * size;
								y_Max = frame_detect.rows - 1 * size;
								break;
							}
							}
							//cout << endl << "特征序号 " << type << endl;
							for (rect.x = 0; rect.x <= x_Max; rect.x++)
							{
								for (rect.y = 0; rect.y <= y_Max; rect.y++)
								{
									//cout << rect.x << "|" << rect.y << "   ";
									m_feature.push_back(HaarFeature(rect, type, size));
								}
							}
						}
					}


					float m_feat;
					int feat_num = (int)m_feature.size();


					//m_feat = m_feature[3].caluHf(frame_detect);

					for (int i = 0; i < feat_num; i++)
					{
						m_feat = m_feature[i].caluHf(frame_detect);
						//cout << i << "|";
						//cout << m_feat << endl;
						if (m_feat < -1.70141e+38 || m_feat > 1.70141e+38)
						{
							cout << i << endl;
						}
						test_data.at<float>(0, i) = m_feat;
					}


					float result = rtree.predict(test_data, Mat());


					if (fabs(result - 1) <= FLT_EPSILON)
					{
						_car.push_back(pos);
					}

			}
		}



}