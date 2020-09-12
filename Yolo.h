#pragma once
#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


typedef struct YoloDetSt {
	std::string label;
	float confidences;
	cv::Rect rect;
}YoloDetSt;

class Yolo {
public:
	Yolo(std::string& modelPath, std::string& configPath, std::string& classesFile, bool isGpu);
	~Yolo();
	int loadModel();
	int runningYolo(cv::Mat& img, std::vector<YoloDetSt>& yoloRet);
	void drowBoxes(cv::Mat& img, std::vector<YoloDetSt>& yoloRet);
	void saveVider(cv::Mat img, std::vector<YoloDetSt>& yoloRet);
private:
	std::string m_modelPath;
	std::string m_configPath;
	std::string m_classesFile;
	std::vector<std::string> m_outNames;
	bool m_isGpu = false;
	std::vector<std::string> m_classes;
	cv::dnn::Net m_net;

	// Yolo参数设置
	float m_confThreshold = 0.5;
	float m_nmsThreshold = 0.4;
	float m_scale = 0.00392;
	cv::Scalar m_mean = { 0,0,0 };
	bool m_swapRB = true;
	int m_inpWidth = 416;
	int m_inpHeight = 416;

	// 检测的图片保存成视频的参数
	int m_saveH = 0;
	int m_saveW = 0;
	cv::VideoWriter m_viderWriter;
	std::string m_viderName;
	int m_frames = 0;

	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<YoloDetSt>& yoloRet);
	std::string getLocNameTime();                    // 返回格式化时间：20200426_150925
	void setViderWriterPara(const cv::Mat& img);
};