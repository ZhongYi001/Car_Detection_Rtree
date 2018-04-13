#pragma once
#include <opencv2/opencv.hpp>  
#include <vector>  

using namespace cv;

class HaarFeature
{
public:
	HaarFeature(const int& size_min, const int& size_max, const int& step = 1, const int& pic_size = 24);
	//~HaarFeature();  // ±¨´í£¬É¾µô

	float caluHf(Mat& _img, vector<float>& m_feat);

private:
	std::vector<Rect> m_rects;
	std::vector<float> m_weights;
	//std::vector<int> m_size;

	float sum(Mat& _imgInt, Rect& rect);
};