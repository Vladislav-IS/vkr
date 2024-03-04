#include "dataloader.h"

using namespace cv;
using namespace std;

DataLoader::DataLoader()
{
    video_capture.open(0, cv::CAP_DSHOW);
    video_capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    QDir dir(dir_path);
    QFile csv(csv_file_name);
    if (!dir.exists())
    {
        dir.mkdir(".");
        csv.open(QIODevice::WriteOnly | QIODevice::Text);
        csv.write("jpg_name,x_norm,y_norm\n");
        csv.close();
    }
    else
    {
        csv.open(QIODevice::ReadOnly | QIODevice::Text);
        lastIndex = QString(csv.readAll()).split("\n").size()-2;
        csv.close();
    }
    timer = new QTimer(this);
    connect(timer,&QTimer::timeout,this,&DataLoader::getFrameOnTimer);
    timer->start(10);
}

void DataLoader::getFrameOnTimer()
{
    //qApp->processEvents();
    //qDebug() << "getFrameOnTimer";
    video_capture.read(frame);
}

void DataLoader::getFrameAndCoords(double mouseX, double mouseY)
{
    //qApp->processEvents();
    //timer->stop();
    video_capture.read(frame);
    //qDebug() << "getFrameAndCoords" << frame.size().width << frame.size().height;
    QString jpg_name = QString::number(QDateTime::currentMSecsSinceEpoch())+".jpg";
    jpg_file_names.append(dir_path+"//"+jpg_name);
    imwrite((dir_path+"//"+jpg_name).toStdString(), frame);
    QFile csv(csv_file_name);
    csv.open(QIODevice::Append | QIODevice::Text);
    csv.write((jpg_name+","+QString::number(mouseX)+","+QString::number(mouseY)+"\n").toStdString().c_str());
    //timer->start(10);
}

void DataLoader::deleteLast()
{
    if (!jpg_file_names.isEmpty())
    {
        QFile delFile(jpg_file_names.last());
        jpg_file_names.removeLast();
        delFile.remove();
        QFile csv(csv_file_name);
        csv.open(QIODevice::ReadWrite | QIODevice::Text);
        QStringList str_list = QString(csv.readAll()).split("\n");
        qDebug() <<str_list;
        str_list = str_list.mid(0,1+lastIndex+jpg_file_names.size());
        //qDebug() <<str_list << 1+lastIndex+jpg_file_names.size();
        csv.write("");
        csv.close();
        csv.open(QIODevice::WriteOnly | QIODevice::Text);
        QString update = str_list.join("\n")+"\n";
        csv.write(update.toStdString().c_str());
        csv.close();
    }
}

DataLoader::~DataLoader()
{
    video_capture.release();
}
