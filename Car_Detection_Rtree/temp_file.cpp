//CvSize size = cvSize(128, 128);			// Cvsize r = cvSize(int width, int height)
//double time0 = static_cast<double>(getTickCount());
//Getting_Haar_From_frame(dframe, 1, size);
//time0 = ((double)getTickCount() - time0) / getTickFrequency();
//cout << "提取特征运行时间为：" << time0 << "秒" << endl;
//
//
//cout << "Done Getting Haar!" << endl;
//
//
//vector<Rect> car;
//Mat detect_data = Mat(All_Haar.size(), 5859, CV_32FC1);			// 特征集Mat
//Mat detect_class = Mat(All_Haar.size(), 1, CV_32FC1);		// 标签矩阵
//double result;
//if (read_Haar_from_vector(detect_data, All_Haar, detect_class))
//{
//	Mat detect_sample;
//	for (int dsample = 0; dsample < All_Haar.size(); dsample++)
//	{
//		detect_sample = detect_data.row(dsample);
//		result = rtree->predict(detect_sample, Mat());
//		//printf("Testing Sample %i -> class result (digit %d)\n", dsample, (int) result);
//		// (N.B. openCV uses a floating point decision tree implementation!)  
//		if (fabs(result - 1) <= FLT_EPSILON)
//		{
//			int x = window_position[4 * dsample];
//			int y = window_position[4 * dsample + 1];
//			int width = window_position[4 * dsample + 2];
//			int height = window_position[4 * dsample + 3];
//			// if they differ more than floating point error => wrong class  
//			car.push_back(windowrect[dsample]);
//			cout << dsample << ":" << x << "|" << y << "|" << width << "|" << height << endl;
//		}
//		else
//		{
//
//		}
//	}
//}
//
//for (size_t i = 0; i<car.size(); i++)
//{
//	//if ((car[i].height > 80) && (car[i].width > 80))
//	{
//		/*	cout << car[i].height << " " << car[i].width << endl;*/
//		rectangle(frame, car[i], Scalar(255, 0, 255), 2);
//	}
//	//else
//	//{
//	//	continue;
//	//}
//}
//imshow(window_name, frame);