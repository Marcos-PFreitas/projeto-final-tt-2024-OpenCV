#ifndef FRAMEPROCESSER_HPP
#define FRAMEPROCESSER_HPP
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

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace std;

class FrameProcesser
{
public:
    // Funcao que faz o processamento dos frames obtidos
    void processFrame(Mat &frame, Net &net, Ptr<cv::face::Facemark> facemark)
    {
        CalculateAR calculateAR;
        int down_width = 900;
        double aspect_ratio = static_cast<double>(frame.cols) / frame.rows;
        int down_height = static_cast<int>(down_width / aspect_ratio);
        resize(frame, frame, Size(down_width, down_height), 0.5, 0.5, INTER_LINEAR); // Diminuiu a resolução para 50% para manter um padrao de imagem

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // auto x = CalculateDNN::detectFace
        
        std::vector<cv::Rect> faces = CalculateDNN::detectFace(net, frame);
        
        // auto x = CalculateDNN();
        // auto _net = x.getInstance();

        std::vector<std::vector<Point2f>> landmarks;
        for (const auto &face : faces)
        {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, "Rosto", cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
        }

        if (facemark->fit(frame, faces, landmarks))
        {
            for (size_t i = 0; i < faces.size(); i++)
            {
                if (landmarks[i].size() >= 68)
                {
                    drawLandmarks(frame, landmarks[i]);

                    double leftEAR = calculateAR.eyes(landmarks[i], 36);
                    double rightEAR = calculateAR.eyes(landmarks[i], 42);
                    double ear = (leftEAR + rightEAR) / 2.0;

                    const double EAR_THRESHOLD = 0.25;
                    double mar = calculateAR.mouth(landmarks[i]);
                    const double MAR_THRESHOLD = 0.6;

                    bool mouthOpen = mar > MAR_THRESHOLD;

                    cv::Point p1(landmarks[i][48].x, landmarks[i][50].y);
                    cv::Point p2(landmarks[i][54].x, landmarks[i][57].y);

                    cv::rectangle(frame, p1, p2, Scalar(255, 55, 0), 2);

                    const std::string status = (ear < EAR_THRESHOLD) ? "Olhos Fechados" : "Olhos Abertos";
                    const std::string mouthStatus = mouthOpen ? "Boca Aberta" : "Boca Fechada";

                    // Reposicionar texto na parte inferior do frame
                    int baseLine = 0;
                    Size textSizeStatus = getTextSize(status, FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseLine);
                    Size textSizeMouth = getTextSize(mouthStatus, FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseLine);

                    int textYPos = frame.rows - 10; // 10 pixels acima da parte inferior do frame
                    int textXPosStatus = (frame.cols - textSizeStatus.width) / 2;
                    int textXPosMouth = (frame.cols - textSizeMouth.width) / 2;

                    putText(frame, status, Point(textXPosStatus, textYPos - 30), FONT_HERSHEY_SIMPLEX, 0.9,
                            (ear < EAR_THRESHOLD) ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);

                    putText(frame, mouthStatus, Point(textXPosMouth, textYPos), FONT_HERSHEY_SIMPLEX, 0.9,
                            mouthOpen ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
                }
                else
                {
                    putText(frame, "Landmarks Insuficientes", Point(faces[i].x, faces[i].y - 10),
                            FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 0), 2);
                }
            }
        }

        imshow("Resultado", frame);
    }
};

#endif // FRAMEPO_HPP