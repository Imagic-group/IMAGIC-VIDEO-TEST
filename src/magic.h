#ifndef MAGIC_H

#define MAGIC_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <vector>

namespace IMAGIC
{
    double dist(const cv::Vec3b& a, const cv::Vec3b& b);
    std::vector<cv::Vec3b> get_keys(const cv::Mat& im);
    void remove_treshold(cv::Mat& im, int rude);
    cv::Mat get_mask(const cv::Mat im, const std::vector<cv::Vec3b>& keys, int sensivity);
    void solve(cv::Mat& im, cv::Mat& bg, const cv::Mat& mask0, const cv::Mat& mask1);
    void equalize(cv::Mat& im);
    void fit(const cv::Mat& a, cv::Mat& b);
    cv::Mat ChromaKey(int quality, cv::Mat im, cv::Mat bg, int sensivity = -1);
}

double IMAGIC::dist(const cv::Vec3b& a, const cv::Vec3b& b)
{
    return sqrt(
    (a[0] - b[0]) * (a[0] - b[0]) +
    (a[1] - b[1]) * (a[1] - b[1]) +
    (a[2] - b[2]) * (a[2] - b[2])
    );
}

std::vector<cv::Vec3b> IMAGIC::get_keys(const cv::Mat& im)
{
    int n = im.rows;
    int m = im.cols;
    return {
    im.at<cv::Vec3b>(0, 0),
    im.at<cv::Vec3b>(0, m - 1),
    im.at<cv::Vec3b>(n - 1, 0),
    im.at<cv::Vec3b>(n - 1, m - 1)
    };
}

void IMAGIC::remove_treshold(cv::Mat& im, int rude)
{
    int lowThreshold = 100;
    int ratio = 3;
    int kernelSize = 3;
    
    cv::Mat cannyEdges, blurred;
  
    if (rude < 3)
    {
        erode(im, im, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    }
    
    blur(im, cannyEdges, cv::Size(3, 3));
    Canny(cannyEdges, cannyEdges, lowThreshold, lowThreshold * ratio, kernelSize);
    
    int dilation_size = rude;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
    cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
    cv::Point(dilation_size, dilation_size));
    
    dilate(cannyEdges, cannyEdges, element);
    im.copyTo(blurred);
    blur(blurred, blurred, cv::Size(3, 3));
    blurred.copyTo(im, cannyEdges);
}

cv::Mat IMAGIC::get_mask(const cv::Mat im, const std::vector<cv::Vec3b>& keys, int sensivity)
{
    double dst = 0;
    for (auto key1 : keys)
    {
        for (auto key2 : keys)
        {
            dst = std::max(dst, dist(key1, key2));
        }
    }
    //Mat res(im.size(), CV_8UC1, Scalar::all(255));
  
    //ADDED {  
    if (sensivity == -1)
    {
        sensivity = 64;
    }
    cv::Vec3b mn = keys[0], mx = keys[0];
    for (cv::Vec3b key : keys)
    {
        for (int i = 0; i < 3; ++i)
        {
            mn[i] = std::min(int(mn[i]), std::max(0, int(key[i]) - 50));
            mx[i] = std::max(int(mx[i]), std::min(int(key[i]) + 50, 255));
        }
    }
    cv::Mat res, tmp(im.size(), CV_8UC1, cv::Scalar::all(255));
    inRange(im, mn, mx, res);
    tmp -= res;
    return tmp;
    //} ADDED
  
    /*
    for (auto key : keys)
    {
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < im.rows; ++i)
        {
            for (int j = 0; j < im.cols; ++j)
            {
                if (sensivity == -1)
                {
                    if (dist(im.at<Vec3b>(i, j), key) < max(64, int(dst)))
                    {
                        res.at<unsigned char>(i, j) = 0;
                    }
                }
                else if (dist(im.at<Vec3b>(i, j), key) < sensivity)
                {
                    res.at<unsigned char>(i, j) = 0;
                }
            }
        }
    }
    */
    return res;
}

void IMAGIC::solve(cv::Mat& im, cv::Mat& bg, const cv::Mat& mask0, const cv::Mat& mask1)
{
    im.convertTo(im, CV_32FC3);
    bg.convertTo(bg, CV_32FC3);
  
    cv::Mat alpha;
    merge(std::vector<cv::Mat>{mask0, mask1, mask0}, alpha);
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);

    cv::Mat res = cv::Mat::zeros(im.size(), im.type());
    multiply(alpha, im, im); 
    multiply(cv::Scalar::all(1.0)-alpha, bg, bg); 
    add(im, bg, res);

    res.convertTo(res, CV_8UC3, 1);
    swap(res, im);
}

void IMAGIC::equalize(cv::Mat& im)
{
    cv::Mat hist_equalized_image;
    cvtColor(im, hist_equalized_image, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> vec_channels;
    split(hist_equalized_image, vec_channels); 

    equalizeHist(vec_channels[0], vec_channels[0]);

    merge(vec_channels, hist_equalized_image); 
      
    cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
    swap(im, hist_equalized_image);
}

void IMAGIC::fit(const cv::Mat& a, cv::Mat& b)
{
    double aa = double(a.rows) / a.cols;
    double bb = double(b.rows) / b.cols;
  
    double scale;
    if (aa > bb)
    {
        scale = double(a.rows) / b.rows;
    }
    else
    {
        scale = double(a.cols) / b.cols;
    }
    /*
    int startY = (-a.rows / 2) / scale + b.rows / 2,
        startX = (-a.cols / 2) / scale + b.cols / 2,
        width  = a.cols
        height = a.rows;

    Mat ROI(b, Rect(startX,startY,width,height));
    ROI.copyTo(b);
    */
    cv::Mat res = cv::Mat::zeros(a.size(), a.type());
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            res.at<cv::Vec3b>(i, j) =
            b.at<cv::Vec3b>((i - a.rows / 2) / scale + b.rows / 2, (j - a.cols / 2) / scale + b.cols / 2);
        }
    }
    swap(b, res);
}

// sensivity = -1 means maximum of (64) and (corner dif)
cv::Mat IMAGIC::ChromaKey(int quality, cv::Mat im, cv::Mat bg, int sensivity)
{
    if (im.size() != bg.size())
    {
        fit(im, bg);
    }

    cv::Mat for_mask = im.clone();
    //equalize(for_mask);
    if (quality == 1)
    {
        bilateralFilter(im, for_mask, 3, 150, 150); // Size(5, 5), 150
        //adaptiveBilateralFilter(im, for_mask, Size(3, 3), 150); // Size(5, 5), 150
    }
    std::vector<cv::Vec3b> keys = get_keys(for_mask);
    cv::Mat mask = get_mask(for_mask, keys, sensivity);

    if (quality == 1)
    {
        cv::Mat mask1 = mask.clone();
        #pragma omp parallel sections   
            #pragma omp section
                remove_treshold(mask, 5); // 5
            #pragma omp section
                remove_treshold(mask1, 3); // 2
    
        solve(im, bg, mask, mask1);
    }
    else
    {
        solve(im, bg, mask, mask);
    }
    return im;
}

#endif /* MAGIC_H */