#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qpixmap.h>
#include <qactiongroup.h>
#include <qtextbrowser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <time.h>


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void newActionSlot();   //�½�����
    void openActionSlot();  //�򿪲���
    void saveActionSlot();  //���Ϊ����
    void encodeButtonSlot();    //�������
    void decodeButtonSlot();    //�������

private:
    Ui::MainWindow ui;

    //��ǰ��ʾ�ļ�
    cv::Mat Image;  //�򿪵�ͼ��
    int Rows, Cols; //ͼƬ�ĸߺͿ�
    std::vector<uchar> compressedData;   //�Ҷ�ͼ������ļ�����

    float ASize;

    void ReadFile(QString filename);
    void WriteFile(QString filename);

    QLabel* imageLabel;
    QTextBrowser* textBrowser;

    //Action��ѡ��
    QActionGroup* flagGroup;
};





