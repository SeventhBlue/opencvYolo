#include <fstream>
#include "Yolo.h"

Yolo::Yolo(std::string& modelPath, std::string& configPath, std::string& classesFile, bool isGpu) {
	m_modelPath = modelPath;
	m_configPath = configPath;
	m_classesFile = classesFile;
	m_isGpu = isGpu;
}

Yolo::~Yolo() {
	m_viderWriter.release();
}

int Yolo::loadModel() {
	int backendId;
	int targetId;
	// cpu or gpu
	if (m_isGpu) {
		backendId = cv::dnn::DNN_BACKEND_CUDA;
		targetId = cv::dnn::DNN_TARGET_CUDA;
	}
	else {
		backendId = cv::dnn::DNN_BACKEND_OPENCV;
		targetId = cv::dnn::DNN_TARGET_CPU;
	}

	// Open file with classes names.
	if (!m_classesFile.empty()) {
		std::ifstream ifs(m_classesFile.c_str());
		if (!ifs.is_open()) {
			std::string error = "File " + m_classesFile + " not found";
			std::cout << error << std::endl;
			return -1;
		}
		std::string line;
		while (std::getline(ifs, line)) {
			m_classes.push_back(line);
		}
	}

	// Load a model.
	m_net = cv::dnn::readNet(m_modelPath, m_configPath);
	m_net.setPreferableBackend(backendId);
	m_net.setPreferableTarget(targetId);

	m_outNames = m_net.getUnconnectedOutLayersNames();

	return 0;
}

int Yolo::runningYolo(cv::Mat& img, std::vector<YoloDetSt>& yoloRet) {
	// Create a 4D blob from a frame.
	cv::Mat blob;
	cv::Mat frame;
	cv::Size inpSize(m_inpWidth > 0 ? m_inpWidth : img.cols,
		m_inpHeight > 0 ? m_inpHeight : img.rows);
	cv::dnn::blobFromImage(img, blob, m_scale, inpSize, m_mean, m_swapRB, false);

	// Run a model.
	m_net.setInput(blob);
	if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		cv::resize(img, img, inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		m_net.setInput(imInfo, "im_info");
	}
	std::vector<cv::Mat> outs;
	m_net.forward(outs, m_outNames);
	postprocess(img, outs, m_net, yoloRet);
	return 0;
}

void Yolo::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<YoloDetSt>& yoloRet) {
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7) {
			float confidence = data[i + 2];
			if (confidence > m_confThreshold) {
				int left = (int)data[i + 3];
				int top = (int)data[i + 4];
				int right = (int)data[i + 5];
				int bottom = (int)data[i + 6];
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "DetectionOutput") {
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7) {
			float confidence = data[i + 2];
			if (confidence > m_confThreshold) {
				int left = (int)(data[i + 3] * frame.cols);
				int top = (int)(data[i + 4] * frame.rows);
				int right = (int)(data[i + 5] * frame.cols);
				int bottom = (int)(data[i + 6] * frame.rows);
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "Region") {
		for (size_t i = 0; i < outs.size(); ++i) {
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > m_confThreshold) {
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}
	}
	else {
		std::cout << "Unknown output layer type: " + outLayerType << std::endl;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];

		std::string label;
		if (!m_classes.empty()) {
			CV_Assert(classIds[idx] < (int)m_classes.size());
		}
		yoloRet.push_back(YoloDetSt{ m_classes[classIds[idx]], confidences[idx], boxes[idx] });
	}
}

void Yolo::drowBoxes(cv::Mat& img, std::vector<YoloDetSt>& yoloRet) {
	for (int i = 0; i < yoloRet.size(); i++) {
		cv::rectangle(img, yoloRet[i].rect, cv::Scalar(0, 0, 255));
		std::string label = cv::format("%.2f", yoloRet[i].confidences);
		label = yoloRet[i].label + ": " + label;
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int top = cv::max(yoloRet[i].rect.y, labelSize.height);
		rectangle(img, cv::Point(yoloRet[i].rect.x, top - labelSize.height),
			cv::Point(yoloRet[i].rect.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
		putText(img, label, cv::Point(yoloRet[i].rect.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	}
}

// 返回格式化时间：20200426_150925
std::string Yolo::getLocNameTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_name = cv::format("%d", t.tm_year + 1900) + cv::format("%.2d", t.tm_mon + 1) + cv::format("%.2d", t.tm_mday) + "_" +
		cv::format("%.2d", t.tm_hour) + cv::format("%.2d", t.tm_min) + cv::format("%.2d", t.tm_sec);
	return time_name;
}

void Yolo::setViderWriterPara(const cv::Mat& img) {
	m_saveH = img.size().height;
	m_saveW = img.size().width;
	m_viderName = "./data/" + getLocNameTime() + ".avi";
	m_viderWriter = cv::VideoWriter(m_viderName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(m_saveW, m_saveH));
}

void Yolo::saveVider(cv::Mat img, std::vector<YoloDetSt>& yoloRet) {
	drowBoxes(img, yoloRet);
	if ((m_saveH == 0) && (m_saveW == 0)) {
		setViderWriterPara(img);
		m_viderWriter << img;
	}
	else {
		if ((m_saveH != img.size().height) || (m_saveW != img.size().width)) {
			cv::resize(img, img, cv::Size(m_saveW, m_saveH));
			m_viderWriter << img;
		}
		else {
			m_viderWriter << img;
		}
	}

	++m_frames;
	if (m_frames == 25 * 60 * 10) {   // 每十分钟从新录制新视频
		m_saveH = 0;
		m_saveW = 0;
		m_viderWriter.release();
	}
}