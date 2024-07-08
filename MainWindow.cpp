#include "MainWindow.h"

int Entropy1D(cv::Mat src, double& dEntropyValue)
{
    double temp[256] = { 0.0 };

    for (int r = 0; r < src.rows; r++) {
        const uchar* pData = src.ptr<uchar>(r);
        for (int c = 0; c < src.cols; c++) {
            temp[pData[c]]++;
        }
    }

    int iSize = src.cols * src.rows;
    for (int i = 0; i < 256; i++)
        temp[i] /= iSize;

    dEntropyValue = 0;
    for (int i = 0; i < 256; i++)
    {

        if (temp[i] < DBL_EPSILON)
            dEntropyValue = dEntropyValue;
        else
            dEntropyValue -= temp[i] * (log(temp[i]) / log(2.0));
    }

    return 0;
}

void getTextS(const std::vector<uchar>& v, double &dEntropyValue) {
    double temp[256] = { 0.0 };
    for (int i = 0; i < v.size(); ++i) {
        temp[v[i]]++;
    }
    for (int i = 0; i < 256; i++)
        temp[i] /= v.size();

    dEntropyValue = 0;
    for (int i = 0; i < 256; i++)
    {
        dEntropyValue += temp[i] * sizeof(uchar);
    }
}

// 将编码的内容写入bin文件保存
void SaveToBinFile(const std::vector<uchar>& data, int rows, int cols, std::string filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open()) {   //文件成功打开
        //写入高和宽
        file.write(reinterpret_cast<char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<char*>(&cols), sizeof(int));
        //写入容器
        file.write(reinterpret_cast<const char*>(data.data()), sizeof(uchar) * data.size());
        file.close();
    }
}

// 将bin文件还原成编码属性
void ReadFromBinFile(std::vector<uchar>& data, int &rows, int &cols, std::string filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (file.is_open()) {   // 文件成功打开
        //读取高和宽
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));

        //读取容器
        file.seekg(0, std::ios::end);
        std::streampos endPos = file.tellg(); // 获取文件末尾位置
        //设置起始读指针
        file.seekg(2*sizeof(int), std::ios::beg);   //移动到vector的位置
        std::streampos startPos = file.tellg();

        data.resize((endPos - startPos) / sizeof(uchar));
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar));
        file.close();
    }
}

//彩图转换为灰度图
cv::Mat TransGray(const cv::Mat &input) {
    cv::Mat output(input.rows, input.cols, CV_8UC1);
    //灰度值 = 0.299R + 0.587G + 0.114B
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output.at<uchar>(i, j) = input.at<cv::Vec3b>(i, j)[0] * 0.114 +
                input.at<cv::Vec3b>(i, j)[1] * 0.587 + 
                input.at<cv::Vec3b>(i, j)[2] * 0.299;
        }
    }
    return output;
}

// 基于行程编码的图片压缩算法 A,A,A,B -> 3,A,1,B 横向 方法1
std::vector<uchar> RunLengthEncode1(const cv::Mat& image) {
    cv::Mat gray;   //灰度图片
    gray = TransGray(image);
    std::vector<uchar> compressedData;   //一维数组，用来保存RLE编码
    compressedData.push_back(1);    //灰度图片首尾放入1为标记

    //遍历每个像素点
    for (int y = 0; y < gray.rows; ++y) {  //行遍历
        uchar count = 1;    //初始化相同像素点个数
        for (int x = 1; x < gray.cols; ++x) {  //列遍历
            if (gray.at<uchar>(y, x) == gray.at<uchar>(y, x - 1)) {
                count++;
                if (count == 255) { //防止溢出
                    compressedData.push_back(count);
                    compressedData.push_back(gray.at<uchar>(y, x - 1));
                    x++;
                    count = 1;
                }
            }
            else {
                compressedData.push_back(count);
                compressedData.push_back(gray.at<uchar>(y, x - 1));
                count = 1;
            }
        }
        //一行结束时也放入
        compressedData.push_back(count);
        compressedData.push_back(gray.at<uchar>(y, gray.cols - 1));
    }

    return compressedData;  //返回编码存放的容器
}

// 解压缩算法 横向 方法1
cv::Mat RunLengthDecode1(const std::vector<uchar>& compressedData, int rows, int cols) {
    cv::Mat decompressedImage(rows, cols, CV_8UC1); //创建一个矩阵存放解压缩的灰度图片像素点，通道数为1

    int dataIndex = 1;  //读取的下标
    for (int y = 0; y < rows; ++y) {
        int x = 0;
        while (x < cols) {
            uchar count = compressedData[dataIndex++];  //当前像素点的个数
            uchar pixelValue = compressedData[dataIndex++]; //像素点的值
            for (int i = 0; i < count; ++i) {   //3,A -> A,A,A
                decompressedImage.at<uchar>(y, x + i) = pixelValue;
            }
            x += count; //放入了count个像素点
        }
    }

    return decompressedImage;   //返回矩阵(Mat类型)
}

//纵向编码 方法2
std::vector<uchar> RunLengthEncode2(const cv::Mat& image) {
    cv::Mat gray;   //灰度图片
    gray = TransGray(image);

    std::vector<uchar> compressedData;   //一维数组，用来保存RLE编码
    compressedData.push_back(2);    //首位放入2为标记

    //遍历每个像素点
    for (int x = 0; x < gray.cols; ++x) {  //列遍历
        uchar count = 1;    //初始化相同像素点个数
        for (int y = 1; y < gray.rows; ++y) {  //行遍历
            if (gray.at<uchar>(y, x) == gray.at<uchar>(y - 1, x)) {
                count++;
                if (count == 255) { //防止溢出
                    compressedData.push_back(count);
                    compressedData.push_back(gray.at<uchar>(y - 1, x));
                    y++;
                    count = 1;
                }
            }
            else {
                compressedData.push_back(count);
                compressedData.push_back(gray.at<uchar>(y - 1, x));
                count = 1;
            }
        }
        //一列结束时也放入
        compressedData.push_back(count);
        compressedData.push_back(gray.at<uchar>(gray.rows - 1, x));
    }

    return compressedData;  //返回编码存放的容器
}

//横向解码 方法2
cv::Mat RunLengthDecode2(const std::vector<uchar>& compressedData, int rows, int cols) {
    cv::Mat decompressedImage(rows, cols, CV_8UC1); //创建一个矩阵存放解压缩的灰度图片像素点，通道数为1

    int dataIndex = 1;  //读取的下标
    for (int x = 0; x < cols; ++x) {
        int y = 0;
        while (y < rows) {
            uchar count = compressedData[dataIndex++];  //当前像素点的个数
            uchar pixelValue = compressedData[dataIndex++]; //像素点的值
            for (int i = 0; i < count; ++i) {   //3,A -> A,A,A
                decompressedImage.at<uchar>(y + i, x) = pixelValue;
            }
            y += count; //放入了count个像素点
        }
    }

    return decompressedImage;   //返回矩阵(Mat类型)
}

//彩图编码 方法3
std::vector<uchar> RunLengthEncode3(const cv::Mat& image) {
    std::vector<uchar> compressedData;   //一维数组，用来保存RLE编码
    compressedData.push_back(3);    //彩度图片首尾放入3为标记

    //遍历每个像素点的每个通道
    for (int i = 0; i < 3; ++i) {
        for (int y = 0; y < image.rows; ++y) {  //行遍历
            uchar count = 1;    //初始化相同像素点个数
            for (int x = 1; x < image.cols; ++x) {  //列遍历
                if (image.at<cv::Vec3b>(y, x)[i] == image.at<cv::Vec3b>(y, x - 1)[i]) {
                    count++;
                    if (count == 255) { //防止溢出
                        compressedData.push_back(count);
                        compressedData.push_back(image.at<cv::Vec3b>(y, x - 1)[i]);
                        x++;
                        count = 1;
                    }
                }
                else {
                    compressedData.push_back(count);
                    compressedData.push_back(image.at<cv::Vec3b>(y, x - 1)[i]);
                    count = 1;
                }
            }
            //一行结束时也放入
            compressedData.push_back(count);
            compressedData.push_back(image.at<cv::Vec3b>(y, image.cols - 1)[i]);
        }
    }
    
    return compressedData;  //返回编码存放的容器
}

//彩图解码 方法3
cv::Mat RunLengthDecode3(const std::vector<uchar>& compressedData, int rows, int cols) {
    cv::Mat decompressedImage(rows, cols, CV_8UC3); //创建一个矩阵存放解压缩的彩图片像素点，通道数为3

    int dataIndex = 1;  //读取的下标
    for (int i = 0; i < 3; ++i) {
        for (int y = 0; y < rows; ++y) {
            int x = 0;
            while (x < cols) {
                uchar count = compressedData[dataIndex++];  //当前像素点的个数
                uchar pixelValue = compressedData[dataIndex++]; //像素点的值
                for (int j = 0; j < count; ++j) {   
                    decompressedImage.at<cv::Vec3b>(y, x + j)[i] = pixelValue;
                }
                x += count; //放入了count个像素点
            }
        }
    }

    return decompressedImage;   //返回矩阵(Mat类型)
}


//读取文件并显示
void MainWindow::ReadFile(QString filename) {
    //QString判断文件类型
    if (filename.contains(".jpg") || filename.contains(".png") || filename.contains(".bmp")) {  //打开图片文件
        ui.encodeButton->setFixedSize(90, 28);  //根据类型显示按钮
        ui.decodeButton->setFixedSize(0, 0);
        std::string FileName = filename.toStdString();
        if (!compressedData.empty()) {   //删除前一个属性
            compressedData.clear();
            Rows = 0;
            Cols = 0;
            //释放文本框
            delete textBrowser;
            textBrowser = NULL;
        }
        Image = cv::imread(FileName);   //设置图片属性

        //显示图片
        QImage qimg;
        qimg.load(filename);
        imageLabel = new QLabel;
        imageLabel->setScaledContents(true);    //根据label大小缩放图片
        imageLabel->setPixmap(QPixmap::fromImage(qimg));
        //将图片设置为滚动控制
        imageLabel->adjustSize();    //控件适应图像（注意必须放到上一句代码之后）
        ui.scrollArea->setWidget(imageLabel); //设置label为scrollArea的窗帘

        //显示图片大小
        QString text = "Size: ";
        ASize = Image.total() * Image.elemSize() / 1024;    //debug
        text.append(QString::number(Image.total() * Image.elemSize() / 1024));  //将数字转换为字符串
        text.append(" KB");
        ui.sizeLabel->setText(text);

        //打开图片结束
    }
    else {  //打开编码文件
        ui.encodeButton->setFixedSize(0, 0);
        ui.decodeButton->setFixedSize(90, 28);  //显示按钮
        std::string FileName = filename.toStdString();

        //删除前一个属性
        if (!Image.empty()) {
            Image.release();
            //释放图片标签
            delete imageLabel;
            imageLabel = NULL;
        }
                                                       
        //从编码文件中读取属性
        ReadFromBinFile(compressedData, Rows, Cols, FileName);

        //显示编码信息
        QFile* file = new QFile;   //申请一个文件指针
        file->setFileName(filename);   //设置文件路径
        bool ok = file->open(QIODevice::ReadOnly);
        if (ok)
        {
            QTextStream in(file);
            textBrowser = new QTextBrowser;
            textBrowser->setText(in.readAll());    //在标签中显示文件内容
            file->close();
        }
        delete file;
        textBrowser->adjustSize();    //控件适应图像（注意必须放到上一句代码之后）
        ui.scrollArea->setWidget(textBrowser); //设置label为scrollArea的窗帘

        //显示文件大小
        QString text = "Size: ";
        text.append(QString::number(compressedData.size() * sizeof(uchar) / 1024));
        text.append(" KB");
        ui.sizeLabel->setText(text);
    }
}

//保存文件
void MainWindow::WriteFile(QString filename) {
    std::string FileName = filename.toStdString();
    //string判断文件类型
    if (FileName.find(".bin") != std::string::npos)     //保存编码文件
        SaveToBinFile(compressedData, Rows, Cols, FileName);
    else    //保存图片文件
        cv::imwrite(FileName, Image);
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    ui.sizeLabel->setText("Size");
    this->setWindowTitle("新窗口");
    ui.encodeButton->setFixedSize(0, 0);
    ui.decodeButton->setFixedSize(0, 0);
    
    //QAction单选框
    flagGroup = new QActionGroup(this);
    flagGroup->addAction(ui.actionRLE_H);   //灰度图横向遍历
    flagGroup->addAction(ui.actionRLE_V);   //灰度图竖向遍历
    flagGroup->addAction(ui.actionRLE_RGB); //彩图编码
    ui.actionRLE_H->setChecked(true);   //默认灰度横向
    flagGroup->setExclusive(true);


    //连接信号和函数
    connect(ui.newAction, &QAction::triggered, this, &MainWindow::newActionSlot);

    connect(ui.openAction, &QAction::triggered, this, &MainWindow::openActionSlot);

    connect(ui.saveAction, &QAction::triggered, this, &MainWindow::saveActionSlot);

    connect(ui.encodeButton, &QPushButton::clicked, this, &MainWindow::encodeButtonSlot);

    connect(ui.decodeButton, &QPushButton::clicked, this, &MainWindow::decodeButtonSlot);
} 

//创建新窗口
void MainWindow::newActionSlot() {  
    MainWindow* newWindow = new MainWindow(); // 创建新的窗口实例
    newWindow->show(); // 显示新窗口
}

//打开文件
void MainWindow::openActionSlot() { 
    //获取文件路径
    QString filename = QFileDialog::getOpenFileName(this, "Choose a File",
        QCoreApplication::applicationFilePath(), "*.jpg;*.png;*.bmp;*.bin");

    if (filename.isEmpty()) {   //没有打开成功
        QMessageBox::warning(this, "错误", "请选择一个文件!");
        return;
    }
    //打开文件
    ReadFile(filename);

    //打开文件结束
    this->setWindowTitle(filename);

    //创建状态栏显示信息
    QStatusBar* stbar = this->statusBar();
    this->setStatusBar(stbar);  //设置到窗口中
    stbar->showMessage("打开成功!", 3000);    //显示实时信息
}

//另存为
void MainWindow::saveActionSlot() { 
    //判断当前文件是否存在
    if (Image.empty() && compressedData.empty()) {  
        QMessageBox::warning(this, "错误", "There is no file!");
        return;
    }

    //判断当前文件类型
    QString flag = "*.bin";
    if (!Image.empty())   //图片类型
        flag = "*.jpg;*.png;*.bmp";

    //选择保存的路径
    QString filename = QFileDialog::getSaveFileName(this, "Choose a File", 
        QCoreApplication::applicationFilePath(), flag);
    if (filename.isEmpty()) {
        QMessageBox::warning(this, "错误", "请选择一个文件!");
        return;
    }
    //保存文件到路径
    WriteFile(filename);

    //保存结束
    this->setWindowTitle(filename);
    QStatusBar* stbar = this->statusBar();
    this->setStatusBar(stbar);  //设置到窗口中
    stbar->showMessage("保存成功!", 3000);    //显示实时信息
}

//编码函数
void MainWindow::encodeButtonSlot() {
    /*debug
    cv::Mat tmpG = TransGray(Image);
    double testS;
    Entropy1D(tmpG, testS);
    clock_t tbegin = clock();*/
    

    //编码压缩图像 修改属性
    if (ui.actionRLE_H->isChecked())    //横向遍历压缩
        compressedData = RunLengthEncode1(Image);
    else if (ui.actionRLE_V->isChecked())
        compressedData = RunLengthEncode2(Image);
    else if (ui.actionRLE_RGB->isChecked())
        compressedData = RunLengthEncode3(Image);

    /*debug
    clock_t tend = clock();
    float duration = (float)(tend - tbegin) * 1000 / CLOCKS_PER_SEC;
    QMessageBox msgb(this);
    msgb.setText("编码性能如下");
    QString ttt = "编码耗时：";
    ttt.append(QString::number(duration));
    ttt.append("ms\n图片的熵：");
    ttt.append(QString::number(testS));
    getTextS(compressedData, testS);
    ttt.append("\n平均码字长度：");
    ttt.append(QString::number(testS));
    ttt.append("\n图片压缩比：");*/
    
    

    Rows = Image.rows;
    Cols = Image.cols;
    //删掉图像属性
    Image.release();
    delete imageLabel;

    //显示编码信息
    SaveToBinFile(compressedData, Rows, Cols, "TmpData.bin");   //临时文件，用来读取
    QFile* file = new QFile;   //申请一个文件指针
    file->setFileName("TmpData.bin");   //设置文件路径
    bool ok = file->open(QIODevice::ReadOnly);
    if (ok)
    {
        QTextStream in(file);
        textBrowser = new QTextBrowser;
        textBrowser->setText(in.readAll());    //在标签中显示文件内容
        textBrowser->adjustSize();    //控件适应图像（注意必须放到上一句代码之后）
        ui.scrollArea->setWidget(textBrowser); //设置label为scrollArea的窗帘
        file->close();
    }
    delete file;
    remove("TmpData.bin");

    /*ASize = (compressedData.size() * sizeof(uchar) + 2 * sizeof(int)) / 1024 / ASize;
    ttt.append(QString::number(ASize));
    msgb.setInformativeText(ttt);
    msgb.exec();    //debug结束出现弹窗*/

    //显示文件大小
    QString text = "Size: ";
    text.append(QString::number((compressedData.size() * sizeof(uchar) + 2 * sizeof(int)) / 1024));
    text.append(" KB");
    ui.sizeLabel->setText(text);

    //改变按钮
    ui.encodeButton->setFixedSize(0, 0);
    ui.decodeButton->setFixedSize(90, 28);

    //编码结束
    QStatusBar* stbar = this->statusBar();
    this->setStatusBar(stbar);  //设置到窗口中
    stbar->showMessage("编码成功!", 3000);    //显示实时信息
}

//解码函数
void MainWindow::decodeButtonSlot() {
    //根据编码的信息判断编码方法 解码到图片属性
    if (compressedData[0] == 1)
        Image = RunLengthDecode1(compressedData, Rows, Cols);
    else if (compressedData[0] == 2)
        Image = RunLengthDecode2(compressedData, Rows, Cols);
    else if (compressedData[0] == 3)
        Image = RunLengthDecode3(compressedData, Rows, Cols);

    //删除编码属性
    compressedData.clear();
    Rows = Cols = 0;

    //改变按钮
    ui.encodeButton->setFixedSize(90, 28);
    ui.decodeButton->setFixedSize(0, 0);
    
    //显示解码后图片
    cv::imwrite("TmpImage.bmp", Image); //临时图片，用来显示
    ReadFile("TmpImage.bmp");
    remove("TmpImage.bmp");

    //解码结束
    QStatusBar* stbar = this->statusBar();
    this->setStatusBar(stbar);  //设置到窗口中
    stbar->showMessage("解码成功!", 3000);    //显示实时信息
}


MainWindow::~MainWindow()
{}
