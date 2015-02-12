/*
 * ResizeFunctions.cpp
 *
 *  Created on: Feb 11, 2015
 *      Author: agata
 */

#include "boost/filesystem.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

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
//					std::cout << temp_path << std::endl;
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

void ResizeToScale(cv::Mat &temp_image, float val){
	int shorter_leg = (temp_image.size().height > temp_image.size().width) ? temp_image.size().width:temp_image.size().height;

	cv::Mat new_image;
	if(shorter_leg > val){
		double resize_factor = val/(shorter_leg*1.0);
		cv::resize(temp_image, new_image,  cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
		int longer_leg = (new_image.size().height > new_image.size().width) ? new_image.size().width:new_image.size().height;

		int differnce = (longer_leg-256)/2;
		cv::Rect crop_rect;

		if (new_image.size().height > new_image.size().width){
			crop_rect.x = 0;
			crop_rect.y = differnce;
			crop_rect.height = val;
			crop_rect.width = val;
		}
		else{
			crop_rect.x = differnce;
			crop_rect.y = 0;
			crop_rect.height = val;
			crop_rect.width = val;
		}

		temp_image = new_image(crop_rect);
	}
	//temp_image = new_img;
}
void ResizeImagesRandomToScale(fs::path &directory)
{
	std::cout << "yay resizing" <<std::endl;
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
					std::cout << temp_path << std::endl;
					temp_image= cv::imread(temp_path,1);
					//flip(temp_image, temp_image, 2);

					double scale = 128.0/temp_image.rows;
					std::cout << "scale : " << scale << std::endl;
					resize(temp_image, temp_image, cv::Size(), scale, scale, cv::INTER_LINEAR);

					int rand_max_x = temp_image.rows-128;
					int rand_max_y = temp_image.cols-64;

					std::cout << "resized " << rand_max_y <<std::endl;
					//std::cout << rand_max_x << " : " << rand_max_y << std::endl;
					//v2 = rand() % 100 + 1;

					for (int j = 0; j<1; j++){
						int start_x =0;  // v3 in the range 1985-2014
						int start_y = rand() % rand_max_y;
						std::cout << start_x << " : " << start_y << std::endl;
						//std::cout << temp_image.rows << " : " << temp_image.cols << std::endl;

						cv::Rect crop_rect;
						crop_rect.x = start_y;
						crop_rect.y = start_x;
						crop_rect.height = 128;
						crop_rect.width = 64;

						std::cout << crop_rect.x+crop_rect.width << " : " <<temp_image.cols <<std::endl;
						std::cout << crop_rect.y+crop_rect.height << " : " <<temp_image.rows <<std::endl;
						//cv::Mat new_image;
						cv::Mat new_img;
						//new_img= temp_image(crop_rect).clone();
						temp_image(crop_rect).copyTo(new_img);

						fs::path newDir =fs::current_path(); newDir /= "src/train_set/neg_rsz2/";
						std::string new_path = fs::canonical(newDir).string();

						FileNames.push_back(dir_iter->path().filename().string());
						std::vector<std::string>::iterator it = FileNames.begin();
						std::string fn = *it;
						FileNames.pop_back();
						fn = fn.substr(0, fn.rfind("."));
						std::string s = std::to_string(j);
						fn.append(s);
						fn.append(".png");
						new_path.append("/");
						new_path.append(fn);
						std::cout << new_path << std::endl;
						cv::imwrite(new_path, new_img );


					}

				}
			}
		}
	}
}




void ResizeImagesRandom(fs::path &directory)
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
					//flip(temp_image, temp_image, 2);

					int rand_max_x = temp_image.rows-128;
					int rand_max_y = temp_image.cols-64;

					//std::cout << rand_max_x << " : " << rand_max_y << std::endl;
					//v2 = rand() % 100 + 1;

					for (int j = 0; j<1; j++){
						int start_x = rand() % rand_max_x;  // v3 in the range 1985-2014
						int start_y = rand() % rand_max_y;
						//std::cout << start_x << " : " << start_y << std::endl;
						//std::cout << temp_image.rows << " : " << temp_image.cols << std::endl;

						cv::Rect crop_rect;
						crop_rect.x = 3;
						crop_rect.y = 3;
						crop_rect.height = 128;
						crop_rect.width = 64;

						//std::cout << crop_rect.x+crop_rect.width << " : " <<temp_image.cols <<std::endl;
						//std::cout << crop_rect.y+crop_rect.height << " : " <<temp_image.rows <<std::endl;
						//cv::Mat new_image;
						cv::Mat new_img;
						//new_img= temp_image(crop_rect).clone();
						temp_image(crop_rect).copyTo(new_img);

						fs::path newDir =fs::current_path(); newDir /= "src/train_set/neg_rsz2/";
						std::string new_path = fs::canonical(newDir).string();

						FileNames.push_back(dir_iter->path().filename().string());
						std::vector<std::string>::iterator it = FileNames.begin();
						std::string fn = *it;
						FileNames.pop_back();
						fn = fn.substr(0, fn.rfind("."));
						std::string s = std::to_string(j);
						fn.append(s);
						fn.append(".png");
						new_path.append("/");
						new_path.append(fn);
						std::cout << new_path << std::endl;
						cv::imwrite(new_path, new_img );


					}

				}
			}
		}
	}
}


