// FaceDemo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include"seeta_own.h"
#include<Windows.h>
#include "FaceDet.h"
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace seeta_pacakge;


//打开E:/source/test.jpg检测人脸并显示
//打开E:/source/test2.jpg检测到人脸后与test.jpg的人脸进行匹配
//最后会打印这两张脸相似度,注意,如果一张照片有多张脸,这里设置的是只提取最大的一张脸
void test1()
{
	//Mat img = cv::imread("E:/files/sdc/codes/TestImage/0-hu.bmp");
	//Mat img = cv::imread("E:/files/sdc/codes/TestImage/2-peng2.png");
	//Mat img = cv::imread("E:/files/sdc/codes/TestImage/0-duoren.bmp");

	//Mat img = cv::imread("E:/files/sdc/2.png");
	//Mat img = cv::imread("E:/files/sdc/1.jpg");

	Mat img = cv::imread("E:/files/sdc/face_img/c/c3.jpg");


	
	facedector_init();
	facelandmarker_init();
	facerecognizer_init();

	SeetaImageData seeta_img = mat_to_seetaImageData(img);
	SeetaFaceInfoArray face_arry = facedetect(seeta_img);


	//cv::Mat face_crop_img = seetaImageData_to_mat(seeta_img);
	//cv::imwrite("E:/files/sdc/codes/TestImage/face.bmp", face_crop_img);

	if (face_arry.size <= 0)
	{
		cout << "未检测到脸";
		system("pause");
	}

	// 默认 face_arry 按照置信度排序
	sort_by_size(face_arry);

	SeetaRect face_pos = face_arry.data[0].pos;
	vector<SeetaPointF> facemark_vec = get_singleface_mark(seeta_img, face_pos);

	for (int i = 0; i < face_arry.size; i++) {

		std::cout << "size:" << face_arry.data[i].pos.width <<","<< face_arry.data[i].pos.height << std::endl;
		//rectangle(img, Rect(face_arry.data[i].pos.x, face_arry.data[i].pos.y, face_arry.data[i].pos.width, face_arry.data[i].pos.height), Scalar(0, 0, 255));
	}

	for (int i = 0; i < 5; i++)
	{
		//circle(img, Point(facemark_vec.at(i).x, facemark_vec.at(i).y), 5, Scalar(0, 255, 0));
	}
	




	cropface(seeta_img, facemark_vec);

	//cv::Mat face_crop_img = seetaImageData_to_mat(face);
	/*cv::imwrite("E:/files/sdc/codes/TestImage/face.bmp", face_crop_img);*/
	//cv::imshow("1", img);

	float* feature = extract_feature(seeta_img, facemark_vec);


	Mat m2 = cv::imread("E:/files/sdc/codes/TestImage/0-hu2.bmp");
	SeetaImageData sid2 = mat_to_seetaImageData(m2);
	SeetaFaceInfoArray sfia2 = facedetect(sid2);
	sort_by_size(sfia2);
	vector<SeetaPointF> spf2 = get_singleface_mark(sid2, sfia2.data[0].pos);


	float* feature2 = extract_feature(sid2, spf2);
	cout << "大小" << facerecognizer->GetExtractFeatureSize();
	std::cout << "相似度" << facerecognizer->CalculateSimilarity(feature, feature2) << std::endl;
	//for (int j = 0; j < 2048; j++)
	//	std::cout << j << "...." << feature2[j] << std::endl;


	float* f = new float[1024];
	for (int o = 0; o < 1024; o++)
		f[o] = feature[o];

	std::cout << "2相似度" << compare(f, feature2);
	waitKey(0);
}
//打开摄像头,检测眼睛状态并显示以及打印
void test2()
{
	facedector_init();
	facelandmarker_init();
	eyestatedector_init();

	VideoCapture videocapture(0);
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData sid = mat_to_seetaImageData(mat);
		SeetaFaceInfoArray sfia = facedetect(sid);
		sort_by_size(sfia);
		vector<SeetaPointF> spf = get_singleface_mark(sid, sfia.data[0].pos);
		int* eye = eye_state(sid, spf);
		cout << "左眼:" << eye[0] << EYE_STATE_STR[eye[0]]
			<< "   " << "右眼:" << eye[1] << EYE_STATE_STR[eye[1]] << std::endl;
		imshow("1", mat);
		waitKey(1);

	}
}
//读取test.jpg,打开摄像头,分析摄像头的人脸与test.jpg是否相似,并在控制台打印结果
void test3()
{
	facedector_init();
	facelandmarker_init();
	eyestatedector_init();
	facerecognizer_init();

	Mat ori = imread("E:/source/test2.jpg");
	SeetaImageData sid_ori = mat_to_seetaImageData(ori);
	SeetaFaceInfoArray sfia_ori = facedetect(sid_ori);
	sort_by_size(sfia_ori);
	vector<SeetaPointF> spf2 = get_singleface_mark(sid_ori, sfia_ori.data[0].pos);
	float* ori_ss = extract_feature(sid_ori, spf2);

	VideoCapture videocapture(0);
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData sid = mat_to_seetaImageData(mat);
		SeetaFaceInfoArray sfia = facedetect(sid);
		sort_by_size(sfia);
		vector<SeetaPointF> spf = get_singleface_mark(sid, sfia.data[0].pos);
		float* curr = extract_feature(sid, spf);
		std::cout << compare(curr, ori_ss) << std::endl;
		imshow("1", mat);
		waitKey(1);


	}
}

//下面的自己看吧,活体检测,状态检测等等

void test4()
{
	facedector_init();
	facelandmarker_init();
	eyestatedector_init();
	facerecognizer_init();
	faceantspoofing_init();
	VideoCapture videocapture(0);
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData sid = mat_to_seetaImageData(mat);
		SeetaFaceInfoArray sfia = facedetect(sid);
		sort_by_size(sfia);
		vector<SeetaPointF> spf = get_singleface_mark(sid, sfia.data[0].pos);

		int status = predict(sid, sfia.data[0].pos, spf, 1);
		cout << status << SPOOF_STATE_STR[status] << std::endl;
		imshow("1", mat);
		waitKey(1);
	}

}
void test5()
{
	facedector_init();
	facelandmarker_init();
	eyestatedector_init();
	facerecognizer_init();
	faceantspoofing_init();

	VideoCapture videocapture(0);
	facetracker_init(videocapture.get(CAP_PROP_FRAME_WIDTH), videocapture.get(CAP_PROP_FRAME_HEIGHT));
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData sid = mat_to_seetaImageData(mat);
		auto tra = tracker(sid);
		for (int i = 0; i < tra.size(); i++)
		{
			SeetaTrackingFaceInfo id = tra.at(i);
			rectangle(mat, Rect(id.pos.x, id.pos.y, id.pos.width, id.pos.height), Scalar(255, 0, 0));
			putText(mat, to_string(id.PID), Point(id.pos.x, id.pos.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);
		}
		imshow("1", mat);
		waitKey(1);

	}
}
void test6()
{
	facedector_init();
	facelandmarker_init();
	eyestatedector_init();
	facerecognizer_init();
	faceantspoofing_init();
	qualityrelu_init(NoMask);
	VideoCapture videocapture(0);
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData sid = mat_to_seetaImageData(mat);
		SeetaFaceInfoArray sfia = facedetect(sid);
		sort_by_size(sfia);
		vector<SeetaPointF> spf = get_singleface_mark(sid, sfia.data[0].pos);
		imshow("1", mat);
		waitKey(1);
		int status = plot_quality(sid, sfia.data[0].pos, spf);
		std::cout << level_string[status] << std::endl;
	}
}

void test_cls() {
	//Mat img = cv::imread("E:/files/sdc/face_img/c/c.jpg");
	//string ori_path = "E:/files/sdc/face_img/c/c3.jpg";
//	string ori_path = "E:/files/sdc/face_img/2.png";
//	string crop_path = "E:/files/sdc/face_img/c999.jpg";
//
//
//	auto facedet_cls = new FaceDet();
//	Mat mat_img = cv::imread(ori_path.c_str());
//	Mat crop_img;
//
//	float features[1024];
//	SeetaRect rect;
//	facedet_cls->checkimage(mat_img, crop_img, features);
//	cv::imwrite(crop_path.c_str(), crop_img);
}


void test_com() {
	string face_1_path = "E:/files/sdc/face_img/c22.jpg";
	string face_2_path = "E:/files/sdc/face_img/00.jpg";

	Mat face_1 = cv::imread(face_1_path.c_str());
	Mat face_2 = cv::imread(face_2_path.c_str());

	

	float features_1[1024];
	float features_2[1024];

	auto facedet_cls = new FaceDet();

	SeetaImageData f_1 = facedet_cls->mat_to_seetaImageData(face_1);
	SeetaImageData f_2 = facedet_cls->mat_to_seetaImageData(face_2);
	facedet_cls->extract_cropface_feature(f_1, features_1);
	facedet_cls->extract_cropface_feature(f_2, features_2);

	float sim = facedet_cls->calculateSimilarity(features_1, features_1);

	std::cout << "sim = " << sim << std::endl;




	std::ofstream outFile("E:/files/sdc/face_img/floatArray.bin", std::ios::binary);

	// 检查文件是否成功打开
	if (!outFile) {
		std::cerr << "无法打开文件" << std::endl;
		return;
	}

	int fff = sizeof(features_2);
	//outFile.write((char*)(&features_2[0]),  sizeof(features_2));
	outFile.write((char*)(&features_2[0]), sizeof(float) * 1024);


	// 关闭文件
	outFile.close();








	string str ="E:/files/sdc/face_img/floatArray.bin";
	float* freadbufs = new float[1024];
	std::ifstream ifs(str, std::ios::binary);
	int dd = sizeof(freadbufs);
	ifs.read((char*)freadbufs, sizeof(float) * 1024);
	ifs.close();


	float sim2 = facedet_cls->calculateSimilarity(features_1, freadbufs);

	std::cout << "sim2 = " << sim2 << std::endl;



}


void bin_save() {

	// 创建一个float数组
	std::vector<float> floatArray = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

	// 打开文件用于二进制写入
	std::ofstream outFile("floatArray.bin", std::ios::binary);

	// 检查文件是否成功打开
	if (!outFile) {
		std::cerr << "无法打开文件" << std::endl;
		return;
	}

	// 保存数组到文件
	outFile.write(reinterpret_cast<const char*>(floatArray.data()), floatArray.size() * sizeof(float));

	// 关闭文件
	outFile.close();

	return;


}

int main()
{
	//test_cls();
	test_com();
}


