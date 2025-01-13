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

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

const std::string caffeConfigFile = "./DNN_Detection/deploy.prototxt";
const std::string caffeWeightFile = "./DNN_Detection/res10_300x300_ssd_iter_140000_fp16.caffemodel";

// Singleton para o Net
class DNNNetSingleton {
public:
    static Net& getInstance() {
        static Net net;
        if (net.empty()) {
            net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        }
        return net;
    }

private:
    DNNNetSingleton() {}
    ~DNNNetSingleton() {}
    DNNNetSingleton(const DNNNetSingleton&) = delete;
    DNNNetSingleton& operator=(const DNNNetSingleton&) = delete;
};
// Funcao para o calculo de abertura do olho
double calculateEAR(const std::vector<Point2f> &landmarks, int startIndex)
{
    Point2f p1 = landmarks[startIndex];
    Point2f p2 = landmarks[startIndex + 1];
    Point2f p3 = landmarks[startIndex + 2];
    Point2f p4 = landmarks[startIndex + 3];
    Point2f p5 = landmarks[startIndex + 4];
    Point2f p6 = landmarks[startIndex + 5];

    double vertical1 = norm(p2 - p6);
    double vertical2 = norm(p3 - p5);
    double horizontal = norm(p1 - p4);

    return (vertical1 + vertical2) / (1.5 * horizontal);
}

// Funcao para o calculo de abertura da boca
double calculateMAR(const std::vector<Point2f> &landmarks)
{
    Point2f p50 = landmarks[50];
    Point2f p52 = landmarks[52];
    Point2f p48 = landmarks[48];
    Point2f p54 = landmarks[54];
    Point2f p61 = landmarks[61];
    Point2f p67 = landmarks[67];

    double vertical1 = norm(p50 - p67);
    double vertical2 = norm(p52 - p61);
    double horizontal = norm(p48 - p54);

    return (vertical1 + vertical2) / (0.8 * horizontal);
}

// Funcao DNN para a captura do rosto
std::vector<cv::Rect> detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
    std::vector<cv::Rect> faceRects;
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    Mat inputBlob = blobFromImage(frameOpenCVDNN, inScaleFactor, Size(inWidth, inHeight), meanVal, false, false);
    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            faceRects.push_back(Rect(Point(x1, y1), Point(x2, y2)));
        }
    }

    return faceRects;
}

// Funcao que faz o processamento dos frames obtidos
void processFrame(Mat &frame, Net &net, Ptr<Facemark> facemark)
{
    int down_width = 900;
    double aspect_ratio = static_cast<double>(frame.cols) / frame.rows;
    int down_height = static_cast<int>(down_width / aspect_ratio);
    resize(frame, frame, Size(down_width, down_height), 0.5, 0.5, INTER_LINEAR); // Diminuiu a resolução para 50% para manter um padrao de imagem

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces = detectFaceOpenCVDNN(net, frame);
    std::vector<std::vector<Point2f>> landmarks;
    for (const auto &face : faces)
    {
        cv::Mat roiGray = gray(face);
        cv::Mat roiSrc = frame(face);

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

                double leftEAR = calculateEAR(landmarks[i], 36);
                double rightEAR = calculateEAR(landmarks[i], 42);
                double ear = (leftEAR + rightEAR) / 2.0;

                const double EAR_THRESHOLD = 0.25;
                double mar = calculateMAR(landmarks[i]);
                const double MAR_THRESHOLD = 0.6;

                bool mouthOpen = mar > MAR_THRESHOLD;

                cv::Point p1(landmarks[i][48].x, landmarks[i][50].y);
                cv::Point p2(landmarks[i][54].x, landmarks[i][57].y);

                float mouthArea = (p2.x - p1.x) * (p2.y - p1.y);

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

int main()
{
    std::string inputPath = "./New folder/video_calibracao.mp4";
    Net net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
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
        processFrame(frame, net, facemark);
        cv::waitKey(0);
    }
    else
    {
        while (true)
        {
            cap >> frame;
            if (frame.empty())
                break;

            processFrame(frame, net, facemark);

            if (cv::waitKey(1) == 'q')
                break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
