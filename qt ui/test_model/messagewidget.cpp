#include "messagewidget.h"

MessageWidget::MessageWidget(QWidget *parent) : QWidget(parent)
{
    QLabel *label = new QLabel;
    QHBoxLayout *hl = new QHBoxLayout;
    label->setText("Подгрузка модулей. Ожидайте...");
    hl->addStretch();
    hl->addWidget(label);
    hl->addStretch();
    setLayout(hl);
}

void MessageWidget::keyPressEvent(QKeyEvent *event)
{
    if (isVisible() && event->key() == Qt::Key::Key_Escape)
        emit timeToClose();
}
