#pragma once

/*******************
	*距离正则项水平集演化
	*Distance Regularized Level Set Evolution
	*简称Drlse理论
	------------------
	**李纯明教授的matlab源码的c++实现版。
	**并做了一定优化。
********************/
/*******************
	*依赖：
	*OpenCV 3.41
	*开发环境：
	*VS2017
********************/

#include <opencv2/opencv.hpp> 


#ifndef POT

#define SINGLEPOT true
#define DOUBLEPOT false

#endif // !POT


/*
	*!!!!use 32bit-float  --  fast
	建议不适用double，速度很慢，效果不会好很多。
*/
#define FLOAT32DEF


#ifndef FLOAT32DEF

#define FLOATdef double
#define FLOATtype CV_64F

#else

#define FLOATdef float
#define FLOATtype CV_32F

#endif // !FLOAT32DEF



#ifndef AT

#define AT(mat, idx) mat.ptr<FLOATdef>(idx)

#endif

enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,

	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};


class Drlse
{
public:
	Drlse();
	~Drlse();

	Drlse(size_t timestep, size_t iter_inner, FLOATdef lambda, FLOATdef alfa, FLOATdef epsilon, FLOATdef sigma, bool potential);
	
	void init(cv::Mat img, cv::Rect set0);

	//设定感兴趣区域----可以减少计算量
	void init(cv::Mat img, cv::Rect roi, cv::Rect set0);
	void init(cv::Mat img, cv::Rect roi, cv::Mat roiset0);
	void evolution(size_t iternum, int ifshow);

	cv::Mat getres();
	void test();

	//	phi_0, g, lambda,mu, alfa, epsilon, timestep, iter, potentialFunction

private:
	cv::Mat origin;
	//	原图
	cv::Mat phi;
	// 演化轮廓
	cv::Mat g;
	//	梯度矩阵

	size_t roi_x, roi_y;
	size_t roi_h, roi_w;
	size_t iter_inner;
	size_t timestep;

	FLOATdef sigma;
	FLOATdef lambda;
	FLOATdef mu;
	FLOATdef alfa;
	FLOATdef epsilon;

	bool potential;


	void initg(cv::Mat img);
	void initphi(cv::Rect set0);
	void initphi(cv::Mat set0);
	void gradient(cv::Mat src, cv::Mat &Fx, cv::Mat &Fy, char selec);
	void div(cv::Mat nx, cv::Mat ny, cv::Mat &dst);
	void distReg_p2(cv::Mat src, cv::Mat &dst);
	void drlse_edge();
	void Dirac(cv::Mat src, cv::Mat &dst);
	void NeumannBoundCond(cv::Mat src, cv::Mat &dst);
	void del2(cv::Mat src, cv::Mat &dst);
	void show();

	FLOATdef areaspeed(cv::Mat exphi);

	cv::Mat conv2(const cv::Mat &img, const cv::Mat& ikernel, ConvolutionType type);
};


