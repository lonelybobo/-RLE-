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
    void newActionSlot();   //新建操作
    void openActionSlot();  //打开操作
    void saveActionSlot();  //另存为操作
    void encodeButtonSlot();    //编码操作
    void decodeButtonSlot();    //解码操作

private:
    Ui::MainWindow ui;

    //当前显示文件
    cv::Mat Image;  //打开的图像
    int Rows, Cols; //图片的高和宽
    std::vector<uchar> compressedData;   //灰度图像编码文件容器

    float ASize;

    void ReadFile(QString filename);
    void WriteFile(QString filename);

    QLabel* imageLabel;
    QTextBrowser* textBrowser;

    //Action单选组
    QActionGroup* flagGroup;
};





