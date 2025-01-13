#ifndef CALCULATEDNN_HPP
#define CALCULATEDNN_HPP

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

class CalculateDNN
{
private:
    static const size_t inWidth = 300;
    static const size_t inHeight = 300;
    static const double inScaleFactor;
    static const cv::Scalar meanVal;
    static const float confidenceThreshold;

    static const std::string caffeConfigFile;
    static const std::string caffeWeightFile;
    CalculateDNN() {} // Construtor privado


public:
    ~CalculateDNN() {}

    // Deletar o construtor de cópia e o operador de atribuição
    CalculateDNN(const CalculateDNN &) = delete;
    CalculateDNN &operator=(const CalculateDNN &) = delete;

    // Método estático para obter a instância Singleton do Net
    static cv::dnn::Net &getInstance()
    {
        static cv::dnn::Net net;
        if (net.empty())
        {
            net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        }
        return net;
    }

    static std::vector<cv::Rect> detectFace(cv::dnn::Net &net, cv::Mat &frameOpenCVDNN)
    {
        std::vector<cv::Rect> faceRects;
        int frameHeight = frameOpenCVDNN.rows;
        int frameWidth = frameOpenCVDNN.cols;

        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
        net.setInput(inputBlob, "data");
        cv::Mat detection = net.forward("detection_out");

        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidenceThreshold)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

                faceRects.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            }
        }

        return faceRects;
    }
};
const double CalculateDNN::inScaleFactor = 1.0;
const cv::Scalar CalculateDNN::meanVal = cv::Scalar(104.0, 177.0, 123.0);
const float CalculateDNN::confidenceThreshold = 0.7;

const std::string CalculateDNN::caffeConfigFile = "./DNN_Detection/deploy.prototxt";
const std::string CalculateDNN::caffeWeightFile = "./DNN_Detection/res10_300x300_ssd_iter_140000_fp16.caffemodel";
#endif // CALCULATEDNN_HPP