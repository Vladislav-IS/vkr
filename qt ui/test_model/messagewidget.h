#ifndef MESSAGEWIDGET_H
#define MESSAGEWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QHBoxLayout>
#include <QKeyEvent>

class MessageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MessageWidget(QWidget *parent = nullptr);

protected:
    void keyPressEvent(QKeyEvent *event);

signals:
    void timeToClose();
};

#endif // MESSAGEWIDGET_H
