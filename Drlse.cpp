#include "Drlse.h"
#include <vector>
#include <math.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;

//#define Fast

#ifdef Fast
//	查询表优化
//	这里不能这么干。效果惨目忍睹。
#define STEPS 62832	//3.14159*2
FLOATdef* sinTable = NULL;
FLOATdef angstep;

void initSinTable() {
	if (sinTable != NULL)return;

	angstep = CV_PI * 2 / STEPS;
	sinTable = (FLOATdef*)malloc(STEPS * sizeof(FLOATdef));

	FLOATdef ang = 0.0;
	int i = 0;
	do {
		sinTable[i++] = sin(ang);
		ang = ang + angstep;
	} while (ang < CV_PI * 2);
}

FLOATdef fastSin(FLOATdef x) {
	int no = int(x / angstep) % STEPS;
	return sinTable[no];
}
FLOATdef fastCos(FLOATdef x) {
	return fastSin(x + CV_PI / 2);
}

#define TABLEINIT()	initSinTable()

#define sin(x) fastSin(x)
#define cos(x) fastCos(x)
#else
#define TABLEINIT()
#endif
//==========================

Drlse::Drlse()
{
}


Drlse::~Drlse()
{
}

Drlse::Drlse(size_t intimestep, size_t initer_inner, FLOATdef inlambda, FLOATdef inalfa, FLOATdef inepsilon, FLOATdef insigma, bool inpotential) {
	timestep = intimestep;
	iter_inner = initer_inner;
	lambda = inlambda;
	alfa = inalfa;
	epsilon = inepsilon;
	sigma = insigma;

	mu = 0.2 / timestep;

	potential = inpotential;

	TABLEINIT();
}

void Drlse::init(Mat img, Rect inproi, Rect set0) {
	origin = img.clone();
	roi_x = inproi.x;
	roi_y = inproi.y;
	roi_h = inproi.height;
	roi_w = inproi.width;

	Mat imgroi = Mat(
		origin, 
		Range(roi_y, roi_y + roi_h), 
		Range(roi_x, roi_x + roi_w));

	initg(imgroi);
	initphi(set0);

}

void Drlse::init(Mat img, Rect inproi, Mat roiset0) {
	origin = img.clone();
	roi_x = inproi.x;
	roi_y = inproi.y;
	roi_h = inproi.height;
	roi_w = inproi.width;

	Mat imgroi = Mat(
		origin,
		Range(roi_y, roi_y + roi_h),
		Range(roi_x, roi_x + roi_w));

	initg(imgroi);
	initphi(roiset0);

}

void Drlse::init(Mat img, Rect set0) {
	origin = img.clone();
	roi_x = 0;
	roi_y = 0;
	roi_h = origin.rows;
	roi_w = origin.cols;
	initg(img);
	initphi(set0);

	//imshow("g", g);
}

void Drlse::initg(Mat img) {
	size_t H = img.rows, W = img.cols;
	Mat G;
	GaussianBlur(img, G, Size(15, 15), sigma, sigma);

	Mat Gf;
	Mat Ix, Iy;
	G.convertTo(Gf, FLOATtype, 1.0);

	gradient(Gf, Ix, Iy, 'o');

	g = Mat(H, W, FLOATtype);
	for (size_t i = 0; i < H; i++) {
		FLOATdef* gptr = AT(g, i);
		FLOATdef* Ixptr = AT(Ix, i);
		FLOATdef* Iyptr = AT(Iy, i);
		for (size_t j = 0; j < W; j++) {
			FLOATdef f = Ixptr[j] * Ixptr[j] + Iyptr[j] * Iyptr[j];
			gptr[j] = 1 / (f + 1);
		}
	}
}

void Drlse::initphi(Rect set0) {
	size_t H = roi_h, W = roi_w;
	Mat initialLSF(H, W, FLOATtype);
	int z0 = 2;
	initialLSF = z0;

	for (size_t i = set0.y; i < set0.y + set0.height; i++) {
		FLOATdef* initptr = AT(initialLSF, i);
		for (size_t j = set0.x; j < set0.x + set0.width; j++) {
			//initialLSF.at<FLOATdef>(i, j) = -2;
			initptr[j] = -z0;
		}
	}
	phi = initialLSF;
}

void Drlse::initphi(Mat set0) {
	size_t H = roi_h, W = roi_w;
	Mat initialLSF(H, W, FLOATtype);
	int z0 = 2;
	for (size_t i = 0; i < H; i++) {
		FLOATdef* initptr = AT(initialLSF, i);
		uchar* set0ptr = set0.ptr<uchar>(i);
		for (size_t j = 0; j < W; j++) {
			initptr[j] = set0ptr[j] == 255 ? -z0 : z0;
		}
	}
	phi = initialLSF;
}

void Drlse::gradient(Mat src, Mat &dstFx, Mat &dstFy, char selec) {
	size_t H = src.rows, W = src.cols;
	dstFx = Mat(H, W, FLOATtype), dstFy = Mat(H, W, FLOATtype);

	/*
	for (size_t y = 0; y < H; y++) {
		//	水平梯度
		dstFx.at<FLOATdef>(y, 0) = src.at<FLOATdef>(y, 1) - src.at<FLOATdef>(y, 0);
		for (size_t x = 1; x < W - 1; x++) {
			dstFx.at<FLOATdef>(y, x) = (src.at<FLOATdef>(y, x + 1) - src.at<FLOATdef>(y, x - 1)) / 2;
		}
		dstFx.at<FLOATdef>(y, W - 1) = src.at<FLOATdef>(y, W - 1) - src.at<FLOATdef>(y, W - 2);
	}
	*******
	以下为优化代码：
	*/
	
	for (size_t i = 0; i < H; i++) {
		FLOATdef* srcptri = AT(src, i);
		FLOATdef* srcptrf = NULL;
		FLOATdef* srcptrb = NULL;
		if (i > 0) {
			srcptrf = AT(src, i - 1);
		}
		if (i < H - 1) {
			srcptrb = AT(src, i + 1);
		}
		
		FLOATdef* dstFxptr = AT(dstFx, i);
		FLOATdef* dstFyptr = AT(dstFy, i);
		for (size_t j = 0; j < W; j++) {
			if (i == 0) {
				dstFyptr[j] = srcptrb[j] - srcptri[j];
			}
			else if (i == H - 1) {
				dstFyptr[j] = srcptri[j] - srcptrf[j];
			}
			else
			{
				dstFyptr[j] = (srcptrb[j] - srcptrf[j]) / 2;
			}
			if (j == 0) {
				dstFxptr[j] = srcptri[j + 1] - srcptri[j];
			}
			else if (j == W-1) {
				dstFxptr[j] = srcptri[j] - srcptri[j - 1];
			}
			else
			{
				dstFxptr[j] = (srcptri[j + 1] - srcptri[j - 1]) / 2;
			}
		}
	}

}


void Drlse::div(Mat nx, Mat ny, Mat &dst) {
	Mat fx, fy, _;
	gradient(nx, fx, _, 'x');
	gradient(ny, _, fy, 'y');

	dst = fx + fy;
}

void Drlse::Dirac(Mat src, Mat &dst) {
	size_t H = src.rows, W = src.cols;
	FLOATdef sigmad = epsilon;

	dst = Mat(H, W, FLOATtype);
	dst = 0;
	for (size_t i = 0; i < H; i++) {
		FLOATdef* srcptr = AT(src, i);
		FLOATdef* dstptr = AT(dst, i);
		for (size_t j = 0; j < W; j++) {
			if (srcptr[j] <= sigmad && srcptr[j] >= -sigmad) {
				dstptr[j] = (0.5/sigmad)*(1+cos(CV_PI*srcptr[j] /sigmad));
			}
		}
	}
}

void Drlse::NeumannBoundCond(Mat src, Mat &dst) {
	size_t H = src.rows, W = src.cols;
	Mat cop;

	dst = src.clone();

	//	g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]); 

	/*
	dst.at<FLOATdef>(0, 0) = src.at<FLOATdef>(2, 2);
	dst.at<FLOATdef>(H-1, 0) = src.at<FLOATdef>(H-3, 2);
	dst.at<FLOATdef>(0, W-1) = src.at<FLOATdef>(2, W-3);
	dst.at<FLOATdef>(H-1, W-1) = src.at<FLOATdef>(H-3, W-3);
	************
		*以下为优化代码：
	*/
	//FLOATdef* dstptr = dst.ptr<FLOATdef>(0);
	FLOATdef* dstptr = AT(dst, 0);
	FLOATdef* srcptr = AT(src, 2);
	dstptr[0] = srcptr[2];
	dstptr[W - 1] = srcptr[W - 3];

	dstptr = AT(dst, H-1);
	srcptr = AT(src, H-3);
	dstptr[0] = srcptr[2];
	dstptr[W - 1] = srcptr[W - 3];
	
	cop = dst.clone();

	//	g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);    

	FLOATdef* dstptr0 = AT(dst, 0);
	FLOATdef* dstptre = AT(dst, H - 1);
	FLOATdef* copptr2 = AT(cop, 2);
	FLOATdef* copptre = AT(cop, H - 3);

	for (size_t j = 1; j < W-1; j++) {
		dstptr0[j] = copptr2[j];
		dstptre[j] = copptre[j];
	}


	cop = dst.clone();
	//	g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
	for (size_t i = 1; i < H-1; i++) {
		FLOATdef* dstptri = AT(dst, i);
		FLOATdef* copptri = AT(cop, i);
		dstptri[0] = copptri[2];
		dstptri[W - 1] = copptri[W - 3];
	}
}

/*
	实现对应的matlab中的conv2方法。
	抄自一名网友的博客。
*/
Mat Drlse::conv2(const Mat &img, const Mat& ikernel, ConvolutionType type)
{
	Mat dest;
	Mat kernel;
	flip(ikernel, kernel, -1);
	Mat source = img;
	if (CONVOLUTION_FULL == type)
	{
		source = Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;
	filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

	if (CONVOLUTION_VALID == type)
	{
		dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	}
	return dest;
}

/*
	实现matlab中del2方法
*/
void Drlse::del2(Mat src, Mat &dst) {
	IplImage img = IplImage(src);
	IplImage *dstimg = cvCreateImage(Size(src.rows, src.cols), IPL_DEPTH_8U, 1);

	cvLaplace(&img, dstimg);	//	laplance变化

	dst = cvarrToMat(dstimg, true);	//	转换为mat,	true=>是否拷贝数据,	自动管理内存
	cvReleaseImage(&dstimg);
}


void Drlse::distReg_p2(Mat src, Mat &dst) {
	size_t H = src.rows, W = src.cols;
	Mat phi_x, phi_y;
	gradient(src, phi_x, phi_y, 'o');

	Mat dpsx(H, W, FLOATtype);
	Mat dpsy(H, W, FLOATtype);
	for (size_t i = 0; i < H; i++) {
		FLOATdef* xptr = AT(phi_x, i);
		FLOATdef* yptr = AT(phi_y, i);
		FLOATdef* dpsxptr = AT(dpsx, i);
		FLOATdef* dpsyptr = AT(dpsy, i);
		for (size_t j = 0; j < W; j++) {
			FLOATdef s = sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
			FLOATdef ps = 0;
			FLOATdef dps = 0;

			if (s >= 0 && s <= 1) {
				//aptr[j] = 1;
				ps = sin(2 * CV_PI*s) / (2 * CV_PI);
			}
			else if (s > 1) {
				//bptr[j] = 1;
				ps = s - 1;
			}

			dps = s == 0 ? 1 : (ps / s);
			/*s = (s == 0 ? 1 : s);
			ps = (ps == 0 ? 1 : ps);
			dps = ps / s;*/

			dpsxptr[j] = (dps - 1)*xptr[j];
			dpsyptr[j] = (dps - 1)*yptr[j];
		}
	}
	Mat divdst;
	div(dpsx, dpsy, divdst);
	
	Mat del2dst;
	//cvDel2(&src, &del2dst);
	Laplacian(src, del2dst, FLOATtype);
	//del2(src, del2dst);

	dst = divdst + del2dst;

}

void Drlse::drlse_edge() {
	size_t H = phi.rows, W = phi.cols;
	Mat vx, vy;

	FLOATdef smallnumber = 1e-10;
	gradient(g, vx, vy, 'o');

	for (size_t k = 0; k < iter_inner; k++) {
		Mat dst;
		NeumannBoundCond(phi, dst);
		
		Mat phi_x, phi_y;
		gradient(dst, phi_x, phi_y, 'o');

		Mat Nx(H, W, FLOATtype), Ny(H, W, FLOATtype);
		for (size_t i = 0; i < H; i++) {
			FLOATdef* xptr = AT(phi_x, i);
			FLOATdef* yptr = AT(phi_y, i);
			FLOATdef* Nxptr = AT(Nx, i);
			FLOATdef* Nyptr = AT(Ny, i);
			
			for (size_t j = 0; j < W; j++) {
				FLOATdef s = sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
				Nxptr[j] = xptr[j] / (s + smallnumber);
				Nyptr[j] = yptr[j] / (s + smallnumber);
			}

			
		}
		Mat curvature, distRegterm, laplace;
		div(Nx, Ny, curvature);
		if (potential) 
		{


			Laplacian(dst, laplace, FLOATtype);
			//del2(dst, laplace);
			distRegterm =  laplace - curvature;
		}
		else 
		{
			distReg_p2(dst, distRegterm);
		}

		Mat dirac;
		Dirac(dst, dirac);

		/*
		原式：
		Mat areaterm, edgeterm;
		areaterm = dirac.mul(g);

		edgeterm = dirac.mul(Nx.mul(vx) + Ny.mul(vy)) + areaterm.mul(curvature);
		phi = dst + timestep * (mu * distRegterm + lambda * edgeterm + alfa * areaterm);
		*******
		以下为优化代码：
		*/
		for (size_t i = 0; i < H; i++) {
			FLOATdef* phiptr = AT(phi, i);
			FLOATdef* Nxptr = AT(Nx, i);
			FLOATdef* Nyptr = AT(Ny, i);
			FLOATdef* curptr = AT(curvature, i);
			FLOATdef* diracptr = AT(dirac, i);
			FLOATdef* gptr = AT(g, i);
			FLOATdef* vxptr = AT(vx, i);
			FLOATdef* vyptr = AT(vy, i);
			FLOATdef* dstptr = AT(dst, i);
			FLOATdef* distRptr = AT(distRegterm, i);
			for (size_t j = 0; j < W; j++) {
				FLOATdef areaterm = diracptr[j] * gptr[j];
				FLOATdef edgeterm = diracptr[j] * (Nxptr[j] * vxptr[j] + Nyptr[j] * vyptr[j]) + areaterm * curptr[j];
				phiptr[j] = dstptr[j] + timestep * (mu*distRptr[j] + lambda * edgeterm + alfa * areaterm);
			}
		}

	}
}

void Drlse::evolution(size_t iternum, int ifshow) {
	//	iternum =>>	迭代次数
	//	ifshow	=>>	是否显示每次迭代后的图像，为false时将显示处理进度
	if (ifshow > 0) {
		std::cout << std::endl << "正在处理：" << std::endl;
	}
	for (size_t i = 0; i < iternum; i++) {
		if (ifshow > 0) {
			if(ifshow > 1)
				show();
			std::cout << "\r" << (i * 100 / iternum) << "%";
		}

		Mat exphi = phi.clone();
		//	记录前一次迭代结果

		drlse_edge();

		if (areaspeed(exphi) < 0.0005) {
			//	以减少迭代次数达到一定的优化
			//	如无需要可以去掉。
			break;
		}
	}
	if (ifshow > 0) {
		std::cout << "\r" << "100%";
	}
}

FLOATdef Drlse::areaspeed(Mat exphi) {
	//	与上次迭代结果相比得到的扩张速度。
	unsigned int subarea = 0;
	unsigned int area = 0;
	for (int i = 0; i < phi.rows; i++) {
		FLOATdef* phiptr = AT(phi, i);
		FLOATdef* exptr = AT(exphi, i);
		for (int j = 0; j < phi.cols; j++) {
			if (phiptr[j] < 0) {
				area++;
				if (exptr[j] >= 0)subarea++;
			}
		}
	}
	if (area == 0)return 0;
	return subarea / float(area);
}

void Drlse::show() {
	Mat ori;
	cvtColor(origin, ori, CV_GRAY2BGR);
	Mat mask = (phi < 0);
	Mat phid;

	std::vector<std::vector<Point>> contours;
	findContours(mask,
		contours,// 轮廓点    
		RETR_TREE,// 只检测外轮廓    
		CHAIN_APPROX_NONE, // 提取轮廓所有点  
		Point(roi_y, roi_x));

	drawContours(ori, contours, -1, Scalar(0, 0, 255), 1);
	namedWindow("Level Set");
	imshow("Level Set", ori);
	waitKey(1);
}

Mat Drlse::getres() {
	//	得到输出图像。
	Mat res(roi_h, roi_w, CV_8U);
	for (int i = 0; i < roi_h; i++) {
		uchar* resptr = res.ptr<uchar>(i);
		FLOATdef* phiptr = phi.ptr<FLOATdef>(i);
		for (int j = 0; j < roi_w; j++) {
			resptr[j] = phiptr[j] < 0 ? 255 : 0;
		}
	}
	return res;
}

void Drlse::test() {
	using namespace std;
	Mat G;
	Laplacian(origin, G, FLOATtype, 3);
	imshow("22", 16*G);
	/*
		do nothing.
	*/
}

