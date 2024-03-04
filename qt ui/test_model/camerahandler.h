#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <QObject>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <QImage>
#include <QPixmap>
#include <QDebug>

class CameraHandler : public QObject
{
    Q_OBJECT
public:
    explicit CameraHandler(QObject *parent = nullptr);    
    ~CameraHandler();
    cv::Mat captureFrame();
    cv::Mat getFrame();
    void releaseCap();
    bool isDone;

public slots:
    void started();
    void finished();

private:
    cv::VideoCapture cap;
    cv::Mat frame;

signals:
    void sendPixmap(QPixmap);
};

#endif // CAMERAHANDLER_H
