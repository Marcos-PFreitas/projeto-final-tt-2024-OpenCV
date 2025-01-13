#ifndef CALCULATEAR_HPP
#define CALCULATEAR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

class calculateAR
{
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
};

#endif // CALCULATEAR_HPP