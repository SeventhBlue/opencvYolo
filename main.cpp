#include <opencv2/highgui.hpp>

#include "Yolo.h"

using namespace cv;

void runningYoloV3();
void runningYoloV4();


int main(int argc, char** argv)
{
	runningYoloV3();
	//runningYoloV4();

	return 0;
}

void runningYoloV4() {
	String modelPath = "./cfg/yolov4_coco.weights";
	String configPath = "./cfg/yolov4_coco.cfg";
	String classesFile = "./data/coco.names";
	Yolo yolov4 = Yolo(modelPath, configPath, classesFile, true);
	yolov4.loadModel();


	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<YoloDetSt> yoloRet;
		yolov4.runningYolo(frame, yoloRet);
		yolov4.drowBoxes(frame, yoloRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV4 detect results", frame);

		//yolov4.saveVider(frame, yoloRet);
	}
}

void runningYoloV3() {
	String modelPath = "./cfg/yolov3_coco.weights";
	String configPath = "./cfg/yolov3_coco.cfg";
	String classesFile = "./data/coco.names";
	Yolo yolov3 = Yolo(modelPath, configPath, classesFile, true);
	yolov3.loadModel();


	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<YoloDetSt> yoloRet;
		yolov3.runningYolo(frame, yoloRet);
		yolov3.drowBoxes(frame, yoloRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV3 detect results", frame);
	}
}