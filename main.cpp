#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <fann.h>


///// PRZESTRZENIE NAZW //////

using namespace cv;
using namespace std;

///// STALE /////
const float widthToHeight = 4.56f;
const short widthAfter = 20;
const short heightAfter = 30;
const short charsCount = 15;
char arrayOfChars[] = "ADGMERVW0356789";



class possibleChar{
public:
	vector<Point> contour;
	int right, left, up, down;
	int middleX;
	int middleY;

	// konstruktor
	possibleChar(vector<Point> cont) {
		contour = cont;
		right = 0;
		left = 0;
		up = 0;
		down = 0;
		middleX = 0;
		middleY = 0;
	}

	static bool xComp(const possibleChar &a, const possibleChar &b) {
		return a.middleX < b.middleX;
	}

	static bool yComp(const possibleChar &a, const possibleChar &b) {
		return a.middleY < b.middleY;
	}

private:

};

////// HISTOGRAM //////
int drawHistogram(Mat obraz) {
	int histSize = 256; // liczba slupkow, w tym przypadku 256 bo obraz ma wartosci od 0 do 255
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat grey_hist;

	cv::calcHist(&obraz, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(grey_hist, grey_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(grey_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i))),
			Scalar(120, 120, 120), 2, 8, 0);
	}
	namedWindow("Histogram", 0);
	imshow("Histogram", histImage);

	return 0;
}

////// FUNCKJA POROWNUJACA ILOSC ELEMENTOW WEKTOROW /////
bool myComp(vector<int> x, vector<int> y) {
	return x.size() > y.size();
}

///// ZNAJDYWANIE TABLICY METODA SYGNATUR /////
Mat detectPlate(Mat imgOpened) {
	Mat imgWithPlate;
	int cols, rows;
	int current, last;
	int top, bottom;
	bool first = false;

	vector<int> inty;
	vector<vector<int>> vec;
	inty.clear();
	vec.push_back(inty);
	
	rows = imgOpened.rows;
	cols = imgOpened.cols;
	int *signature = new int[rows];


	///// WYZNACZANIE GORNEJ I DOLNEJ GRANICY TABLICY //////
	for (int i = 0; i < rows; i++) {
		last = imgOpened.at<uchar>(i, 0);
		signature[i] = 0;
		for (int j = 0; j < cols; j++) {
			//                            r  c
			current = imgOpened.at<uchar>(i, j);
			if (current != last) { // jesli jest zmiana kontrastu
				signature[i]++;
			}
			last = current;
		}

		if (signature[i] >= 16 && signature[i] <= 40) {
			inty.push_back(i);
		}
		else {
			if (!inty.empty()) {
				vec.push_back(inty);
				inty.clear();
			}
		}
	}

	sort(vec.begin(), vec.end(), myComp);
	bottom = vec[0][0]-15;
	top = vec[0][vec[0].size()-1]+15;

	Rect R(Point(0,top), Point(cols,bottom));
	imgWithPlate = imgOpened(R);

	return imgWithPlate;
}

///// MAKSYMALIZOWANIE KONTRASTU /////
Mat maximizeContrast(Mat imgGray) {
	Mat imgTopHat;
	Mat imgBlackHat;
	Mat imgGrayPlusTopHat;
	Mat imgGrayPlusTopHatMinusBlackHat;

	Mat structuringElement = getStructuringElement(CV_SHAPE_RECT, Size(3, 3));

	morphologyEx(imgGray, imgTopHat, CV_MOP_TOPHAT, structuringElement);
	morphologyEx(imgGray, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

	imgGrayPlusTopHat = imgGray + imgTopHat;
	imgGrayPlusTopHatMinusBlackHat = imgGrayPlusTopHat - imgBlackHat;

	return(imgGrayPlusTopHatMinusBlackHat);
}

///// WYDOBYCIE REJESTRACJI Z OBSZARU KTORY GO POSIADA //////
Mat extractPlate(Mat imgWithPlate) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imgWithPlate, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // RETR_EXTERNAL  RETR_CCOMP
	
	int *counts = new int[contours.size()];

	for (int i = 0; i < contours.size(); i++) 
		counts[i] = 0;

	for (int idx = 0; idx < contours.size(); idx++) {
		if (hierarchy[idx][3]!=-1) {
			counts[hierarchy[idx][3]]++;
		}
	}

	int max = 0;
	int contId = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (counts[i]>max) {
			max = counts[i];
			contId = i;
		}
	}

	int left = imgWithPlate.cols;
	int right = 0;
	for (int i = 0; i < contours[contId].size();i++) {
		if (contours[contId][i].x>right) right = contours[contId][i].x;
		if (contours[contId][i].x < left) left = contours[contId][i].x;
	}

	Mat imgPlate;
	Rect R(Point(left, 0), Point(right, imgWithPlate.rows));
	imgPlate = imgWithPlate(R);

	return imgPlate;
}

vector<Mat> extractChars(Mat imgPlate) {
	Mat ch;
	Rect R;
	vector<Mat> chars;
	vector<vector<Point>> contours;
	vector<possibleChar> possibleChars;
	vector<Vec4i> hierarchy;

	findContours(imgPlate, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // RETR_EXTERNAL  RETR_CCOMP

	for (int idx = 0; idx < contours.size(); idx++) {
		//drawContours(imgWithPlate, contours, idx, Scalar(150), 2, 8, hierarchy);
		if (hierarchy[idx][3] != -1) {
			possibleChars.push_back(possibleChar(contours[idx]));
		}
	}

	if (possibleChars.size()!=7) { //jesli nie mamy 7 znakow (kazda tablica powinna miec 7)
		cout << "Blad przy dzieleniu na znaki" << endl;
		return chars;
	}


	for (int i = 0; i < possibleChars.size();i++) {
		possibleChars[i].up = imgPlate.rows;
		possibleChars[i].down = 0;
		possibleChars[i].left = imgPlate.cols;
		possibleChars[i].right = 0;
		for (int j = 0; j < possibleChars[i].contour.size();j++) {
			if (possibleChars[i].contour[j].x > possibleChars[i].right) possibleChars[i].right = possibleChars[i].contour[j].x;
			if (possibleChars[i].contour[j].x < possibleChars[i].left) possibleChars[i].left = possibleChars[i].contour[j].x;
			if (possibleChars[i].contour[j].y > possibleChars[i].down) possibleChars[i].down = possibleChars[i].contour[j].y;
			if (possibleChars[i].contour[j].y < possibleChars[i].up) possibleChars[i].up = possibleChars[i].contour[j].y;
		}
		possibleChars[i].middleX = (possibleChars[i].up + possibleChars[i].down) / 2;
		possibleChars[i].middleY = (possibleChars[i].left + possibleChars[i].right) / 2;
	}

	sort(possibleChars.begin(), possibleChars.end(), possibleChar::yComp); // sortowanie znakow od lewej do prawej

	for (auto &temp : possibleChars) {
		R = Rect(Point(temp.left, temp.up), Point(temp.right, temp.down));
		ch = imgPlate(R);
		chars.push_back(ch);
		ch.release();
	}

	return chars;
}

char recognizeChar(Mat ch) {
	fann_type vec[widthAfter*heightAfter];
	short current = 0;
	fann_type *calc_out;
	struct fann *ann = fann_create_from_file("plates.net");
	float max = 0.0f;
	short maxId = 0;

	for (int i = 0; i < ch.rows; i++) {
		for (int j = 0; j < ch.cols; j++) {
			vec[current++] = (ch.at<uchar>(i, j)>0) ? 1 : 0;
		}
	}

	calc_out = fann_run(ann, vec);

	for (int k = 0; k < charsCount; k++) {
		if (calc_out[k]>max) {
			max = calc_out[k];
			maxId = k;
		}
		//cout << calc_out[k] << endl;
	}

	fann_destroy(ann);
	return arrayOfChars[maxId];
}

string recognizeChars(vector<Mat> chars) {
	string plate;


	for (auto &ch : chars) {
		plate = plate + recognizeChar(ch);
	}

	return plate;
}

void saveToFile(vector<Mat> chars, char *name) {
	ofstream file(name);
	uchar temp;

	for (auto &ch : chars) {

		for (int i = 0; i < ch.rows; i++) {

			for (int j = 0; j < ch.cols; j++) {
				temp = (ch.at<uchar>(i, j)>0) ? 1 : 0;
				file << (short)temp << " ";
			}

		}
		file << endl;
	}

	file.close();
}

int main(int argc, char *argv[])
{
	Mat imgGray, imgBlured, imgThresh, imgOpened, imgWithPlate, imgPlate;
	vector<Mat> chars;
	string plate;
	string fileName;

	
	if (argc < 2) {
		cout << "Podaj nazwe zdjecia, z ktorego chcesz odczytac numer tablicy rejestracyjnej" << endl;
		cin >> fileName;
	}
	else {
		fileName = argv[1];
	}

	//imgGray = imread("C:\\Users\\7\\Desktop\\inzynierka\\tablice\\4.jpg", CV_LOAD_IMAGE_GRAYSCALE); // CV_LOAD_IMAGE_COLOR
	imgGray = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE); // CV_LOAD_IMAGE_COLOR

	if (imgGray.data==NULL) {
		cout << "Blad przy wczytywaniu zdjecia" << endl;
		return -1;
	}

	if (!imgGray.data) {
		cout << "Wystapil blad przy ladowaniu zdjecia" << endl;
		return -1;
	}

	resize(imgGray, imgGray, Size(), 0.5, 0.5);
	imgGray = maximizeContrast(imgGray);
	GaussianBlur(imgGray, imgBlured, Size(35, 35), 0);
	threshold(imgBlured, imgThresh, 125, 255, THRESH_BINARY);

	Mat structuring_element(7, 7, CV_8U, Scalar(1));
	morphologyEx(imgThresh, imgOpened, MORPH_OPEN, structuring_element);
	
	imgWithPlate = detectPlate(imgOpened); //wykrywanie miejsca z tablica
	imgPlate = extractPlate(imgWithPlate); //wyodrebnienie tablicy
	namedWindow("imgPlate", 0);
	imshow("imgPlate", imgPlate);
	chars = extractChars(imgPlate);

	if (chars.size()!=7) {
		cout << "Nie rozpoznano tablicy" << endl;
		return -1;
	}

	for (int i = 0; i < 7;i++) {
		resize(chars[i], chars[i], Size(widthAfter, heightAfter));
	}

	plate = recognizeChars(chars);
	cout << "Rozpoznana tablica: " << plate << endl;
	waitKey(0);

    return 0;
}

