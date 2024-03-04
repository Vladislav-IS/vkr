#include "camerahandler.h"

CameraHandler::CameraHandler(QObject *parent) : QObject(parent)
{

}

CameraHandler::~CameraHandler()
{

}

void CameraHandler::releaseCap()
{
    isDone = true;
    cap.release();
}

void CameraHandler::started()
{
    //cv::VideoCapture capture(0, cv::CAP_DSHOW);
    //cap = capture;
    isDone = false;
    cap.open(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    while (true)
    {
        //qDebug() << cap.isOpened();
        if (isDone) return;
        cap >> frame;
        if (!frame.empty())
        {
            //cv::imwrite("1.jpg", frame);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            //cv::line(frame,cv::Point(1280/2-1,0),cv::Point(1280/2-1,719),cv::Scalar(0,255,0),4);
            cv::rectangle(frame, cv::Point(457,341-100), cv::Point(457+324,341+361-100),cv::Scalar(0,255,0),4);
            cv::resize(frame, frame, cv::Size(1280/2,720/2));
            QImage img= QImage((uchar*) frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
            emit sendPixmap(QPixmap::fromImage(img));
        }
    }
}

void CameraHandler::finished()
{
    cap.release();
}
