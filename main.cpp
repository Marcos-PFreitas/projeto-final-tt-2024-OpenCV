#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"
#include "CalculateDNN.hpp"
#include "CalculateAR.hpp"
#include "FrameProcesser.hpp"


int main()
{
    FrameProcesser Frame;
    std::string inputPath = "./New folder/video_calibracao.mp4";
    Net net = CalculateDNN::getInstance(); 
    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("./New folder/lbfmodel.yaml");

    cv::Mat frame;
    cv::VideoCapture cap(inputPath);

    if (!cap.isOpened())
    {
        frame = cv::imread(inputPath);
        if (frame.empty())
        {
            std::cerr << "Erro ao abrir o vídeo ou imagem!" << std::endl;
            return -1;
        }
        Frame.processFrame(frame, net, facemark);
        cv::waitKey(0);
    }
    else
    {
        while (true)
        {
            cap >> frame;
            if (frame.empty())
                break;

            Frame.processFrame(frame, net, facemark);

            if (cv::waitKey(1) == 'q')
                break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
