#include "Haar.h"  
#include <iostream>  
#include <math.h>

using namespace std;

HaarFeature::HaarFeature(Rect& rect, int& type, int& size)
{
	//rect = Rect(rect.x, rect.y, 0, 0);
	//for (int size = 1; size <= size_max; size++)
	//{
	switch (type)
	{
	case 0:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 1 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(-1.f);
		m_weights.push_back(2.f);
		m_weights.push_back(0.f);
		m_weights.push_back(0.f);
		break;
	}
	case 1:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(-1.f);
		m_weights.push_back(2.f);
		m_weights.push_back(0.f);
		m_weights.push_back(0.f);
		break;
	}
	case 2:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 3 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 1 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(1.f);
		m_weights.push_back(-3.f);
		m_weights.push_back(3.f);
		m_weights.push_back(0.f);
		break;
	}
	case 3:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 3 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(1.f);
		m_weights.push_back(-3.f);
		m_weights.push_back(3.f);
		m_weights.push_back(0.f);
		break;
	}
	case 4:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 4 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 3 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 1 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(1.f);
		m_weights.push_back(-2.f);
		m_weights.push_back(2.f);
		m_weights.push_back(0.f);
		break;
	}
	case 5:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 4 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 3 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_weights.push_back(1.f);
		m_weights.push_back(-2.f);
		m_weights.push_back(2.f);
		m_weights.push_back(0.f);
		break;
	}
	case 6:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 4 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 2 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 1 * size, rect.height + 4 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 1 * size, rect.height + 1 * size));
		m_weights.push_back(1.f);
		m_weights.push_back(-2.f);
		m_weights.push_back(-2.f);
		m_weights.push_back(4.f);
		break;
	}
	case 7:
	{
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_rects.push_back(Rect(rect.x, rect.y, rect.width + 2 * size, rect.height + 1 * size));
		m_weights.push_back(-1.f);
		m_weights.push_back(0.f);
		m_weights.push_back(0.f);
		m_weights.push_back(0.f);
		break;
	}
	}
	//}
}


float HaarFeature::sum(Mat& _img, Rect& _rect)
{
	int x = _rect.x + _rect.width;
	int y = _rect.y + _rect.height;
	Mat _imgInt;
	int tempVal = 0;
	integral(_img, _imgInt, CV_32F);
	tempVal =
		_imgInt.at<int>(x, y) +
		_imgInt.at<int>(_rect.x + 1, _rect.y + 1) -
		_imgInt.at<int>(x, _rect.y + 1) -
		_imgInt.at<int>(_rect.x + 1, y);
	return tempVal;
}


float HaarFeature::caluHf(Mat& _img)
{
	int Val = 0;
	float a1 = m_weights[0],
		a2 = m_weights[1],
		a3 = m_weights[2],
		a4 = m_weights[3];
	Rect r1 = m_rects[0],
		r2 = m_rects[1],
		r3 = m_rects[2],
		r4 = m_rects[3];
	Val = a1 * sum(_img, r1) +
		a2 * sum(_img, r2) +
		a3 * sum(_img, r3) +
		a4 * sum(_img, r4);

	//float g_sum = sum(_img, Rect(r1.x, r1.y, _img.cols - 2, _img.rows - 2));
	//float g_sqsum = g_sum * g_sum;
	//float mean = g_sum / ((_img.cols - 2) * (_img.rows - 2)),
	//	sqmean = g_sqsum / ((_img.cols - 2) * (_img.rows - 2));
	//float varNormFactor = sqrt(sqmean - mean * mean);


	return Val / (_img.cols * _img.rows);

}