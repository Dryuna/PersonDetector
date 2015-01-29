//============================================================================
// Name        : PersonDetector.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"


namespace fs = boost::filesystem;

void ResizeImages(fs::path &directory);
void DrawGrid(cv::Mat &img_out, int dist, cv::Scalar color, cv::Point offset);

int main() {
	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
    //fs::path someDir =fs::current_path(); someDir /= "src/neg";
    //ResizeImages(someDir);
	fs::path imgDir =fs::current_path(); imgDir /= "src/ws2.png";
	std::string im_loc = fs::canonical(imgDir).string();
	cv::Mat image= cv::imread(im_loc, CV_LOAD_IMAGE_COLOR);
	if( !image.data )  { return -1; }

	cv::Size newsz;
	newsz.width= image.cols/4;
	newsz.height= image.rows/4;
	cv::resize(image, image,newsz );
	cv::Mat im_grey;
	cv::cvtColor(image, im_grey, CV_BGR2GRAY);

	cv::Mat im_xdir;
	float m[3] = {-1,0,1};
	cv::Mat kernel_x(cv::Size(3,1), CV_32F, m);
	cv::Mat kernel_y(cv::Size(1,3), CV_32F, m);
//	kernel_x = cv::Mat::ones(1, 3, CV_32F);
//	kernel_x.data[0]=-1;
//	kernel_x.data[1]=0;
//	kernel_x.data[2]=1;


	cv::filter2D(im_grey, im_xdir, -1 , kernel_y, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT );



	cv::Rect zoom_area= cv::Rect(100,100,32,32);

	cv::Mat zoomed_mat;
	cv::Size zoom_size;
	zoom_size.height = 32*10;
	zoom_size.width =  32*10;
	cv::Mat zoom_in = im_grey(zoom_area);
	cv::resize(zoom_in, zoomed_mat, zoom_size);

	cv::vector<cv::Mat> channels(3);
    channels[0]= zoomed_mat;
    channels[1]= zoomed_mat;
    channels[2]= zoomed_mat;
    cv::Mat zoomed_img;
    cv::merge(channels,zoomed_img);

	//;
    cv::Point fstpt; fstpt.x =0; fstpt.y =0;
    cv::Point sndpt; sndpt.x =80; sndpt.y =80;
	DrawGrid(zoomed_img, 80, cv::Scalar(0,0,255,0), fstpt);
	DrawGrid(zoomed_img, 160, cv::Scalar(255,0,0,0), sndpt);

	std::cout << kernel_y << std::endl;
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Display window", zoomed_img );
	cv::waitKey(15000);

	return 0;
}

void DrawGrid(cv::Mat &img_out, int dist, cv::Scalar color, cv::Point offset){
	//int dist=80;

	int width=img_out.size().width;
	int height=img_out.size().height;

	int x = offset.x;
	int y = offset.y;

	for(int i=0;i<height;i+=dist)
	  cv::line(img_out,cv::Point(x,i),cv::Point(width+x,i),color);

	for(int i=0;i<width;i+=dist)
	  cv::line(img_out,cv::Point(i+y,0),cv::Point(i+y,height),color);

//	for(int i=0;i<width;i+=dist)
//	  for(int j=0;j<height;j+=dist)
//		  img_out.at<cv::Vec4b>(i,j)=color;
}

void ResizeImages(fs::path &directory)
{
	cv::Mat temp_image;

	std::vector<std::string> FileNames;
	fs::directory_iterator end_iter;

	//fs::path someDir(currPath, "/pos");
	if ( fs::exists(directory) && fs::is_directory(directory)){
		for( fs::directory_iterator dir_iter(directory) ; dir_iter != end_iter ; ++dir_iter){
			if (fs::is_regular_file(dir_iter->status()) ){
				if(fs::is_regular_file(*dir_iter) && dir_iter->path().extension() == ".png"){
					//ret.push_back(it->path().filename());


					fs::directory_entry& entry = (*dir_iter);
					std::string temp_path = fs::canonical(entry).string();
					//std::cout << temp_path << std::endl;
					temp_image= cv::imread(temp_path,1);
					int shorter_leg = (temp_image.size().height > temp_image.size().width) ? temp_image.size().width:temp_image.size().height;

					cv::Mat new_image;
					if(shorter_leg > 256){
						double resize_factor = 256.0/(shorter_leg*1.0);
						cv::resize(temp_image, new_image,  cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
						int longer_leg = (new_image.size().height > new_image.size().width) ? new_image.size().width:new_image.size().height;

						int differnce = (longer_leg-256)/2;
						cv::Rect crop_rect;

						if (new_image.size().height > new_image.size().width){
							crop_rect.x = 0;
							crop_rect.y = differnce;
							crop_rect.height = 256;
							crop_rect.width = 256;
						}
						else{
							crop_rect.x = differnce;
							crop_rect.y = 0;
							crop_rect.height = 256;
							crop_rect.width = 256;
						}

						cv::Mat new_img = new_image(crop_rect);

						fs::path newDir =fs::current_path(); newDir /= "src/neg_rsz/";
						std::string new_path = fs::canonical(newDir).string();

						FileNames.push_back(dir_iter->path().filename().string());
						std::vector<std::string>::iterator it = FileNames.begin();
						std::string fn = *it;
						FileNames.pop_back();
						new_path.append("/");
						new_path.append(fn);
						cv::imwrite(new_path, new_img );


					}

				}
			}
		}
	}
}


