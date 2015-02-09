//============================================================================
// Name        : PersonDetector.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"
#include "libsvm-3.20/svm.h"

namespace fs = boost::filesystem;
typedef std::vector<std::string> Filenames;
typedef std::vector<float> FeatureVec;

void GetTrainingData(svm_problem &prob);
void ResizeImages(fs::path &directory);
void ResizeImagesRandom(fs::path &directory);
void DrawGrid(cv::Mat &img_out, int dist, cv::Scalar color, cv::Point offset);
int TestLibSVM();
int TestXOR();
void GetFileNames (Filenames &fn, fs::path &directory);
void GetTrainingData(svm_problem &prob);
void GetTestData(svm_node* testnode, const svm_model *model);
void MakeSparseFeatures (FeatureVec &features, svm_node *x_space);
void GetSparseFeatLength(FeatureVec &features, int &num_l);
void TestData(svm_node* testnode);

int main() {
	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
    //fs::path someDir =fs::current_path(); someDir /= "src/train_set/pos";
    //ResizeImages(someDir);
	//ResizeImagesRandom(someDir);

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
	//GetTrainingData();

	int test = TestLibSVM();
	//int test2 = TestXOR();
//	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
//	cv::imshow( "Display window", zoomed_img );
//	cv::waitKey(15000);

	return 0;
}

int TestXOR(){
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	svm_parameter param;     // set by parse_command_line
	svm_problem prob;        // set by read_problem
	//struct svm_model *model;
	svm_node *x_space;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.5;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;


	//Problem definition-------------------------------------------------------------
	prob.l = 4;

	//x values matrix of xor values
	double matrix[prob.l][2];
	matrix[0][0] = 1;
	matrix[0][1] = 1;

	matrix[1][0] = 1;
	matrix[1][1] = 0;

	matrix[2][0] = 0;
	matrix[2][1] = 1;

	matrix[3][0] = 0;
	matrix[3][1] = 0;


	//This part i have trouble understanding
	svm_node** x = new svm_node *[prob.l];

	//Trying to assign from matrix to svm_node training examples
	for (int row = 0;row <prob.l; row++){
		svm_node* x_space = new svm_node[3];
		for (int col = 0;col < 2;col++){
			x_space[col].index = col;
			x_space[col].value = matrix[row][col];
		}
		x_space[2].index = -1;      //Each row of properties should be terminated with a -1 according to the readme
		x[row] = x_space;
	}

	prob.x = x;

	//yvalues
	prob.y = new double[prob.l];
	prob.y[0] = -1;
	prob.y[1] = 1;
	prob.y[2] = 1;
	prob.y[3] = -1;

	//Train model---------------------------------------------------------------------
	svm_model *model = svm_train(&prob,&param);


	//Test model----------------------------------------------------------------------
	svm_node* testnode = new svm_node[3];
	testnode[0].index = 0;
	testnode[0].value = 1;
	testnode[1].index = 1;
	testnode[1].value = 0;
	testnode[2].index = -1;

	//This works correctly:
	double retval = svm_predict(model,testnode);
	printf("retval: %f\n",retval);


	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	//free(x);

	return 0;
}

int TestLibSVM(){

	//Parameters
	svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 1.0/3780.0;
	param.coef0 = 0;
//	param.nu = 0.5;
	param.cache_size = 1000;
	param.C = 1;
	param.eps = 1e-3;
//	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	svm_problem prob;
	GetTrainingData(prob);
	std::cout << "starting svm_training" << std::endl;
//	svm_model *model = svm_train(&prob,&param);

	//
//	int nr_fold=5;
//	double *target = new double[prob.l];
//	svm_cross_validation(&prob, &param, nr_fold, target);
	const char *model_file_name = "saved_model.model";
//	int saved= svm_save_model(model_file_name, model);
	svm_model *model =svm_load_model(model_file_name);

	svm_node* testnode;
	GetTestData(testnode, model);
//	std::cout << "starting svm_predict" << std::endl;
//	double retval = svm_predict(model,testnode);
//	printf("retval: %f\n",retval);


	double prob_est[2];
	svm_node* testnode2;
	TestData(testnode2);
	std::cout << testnode2->index <<std::endl;
	std::cout << "starting svm_predict_probability" << std::endl;
	double predictions = svm_predict_probability(model, testnode2, prob_est);
	printf("%f\t%f\t%f\n", predictions, prob_est[0], prob_est[1]);

	//int nr_fold=5;
	//double target;
	//svm_cross_validation(&prob, &param, nr_fold, &target);

    svm_destroy_param(&param);
    //free(prob.y);
    //free(prob.x);

	std::cout << "end of demo" << std::endl;

	return 0;

}

void GetTrainingData(svm_problem &prob){
//	cv::Mat test_image;
//	fs::path test_path =fs::current_path(); test_path /= "/src/train_set/pos_rsz/crop_000001a0.png";
//	std::string im_path = fs::canonical(test_path).string();

	Filenames pos_fn, neg_fn;
	fs::path pos_path =fs::current_path(); pos_path /= "/src/train_set/pos_rsz";
	fs::path neg_path =fs::current_path(); neg_path /= "/src/train_set/neg_rsz";
	GetFileNames (pos_fn, pos_path);
	GetFileNames (neg_fn, neg_path);

	std::cout << pos_fn.size() << std::endl;
	std::cout << neg_fn.size() << std::endl;

	//Merge all filenames together into new vector
	Filenames all_fn;
	all_fn.reserve( pos_fn.size() + neg_fn.size() ); // preallocate memory
	all_fn.insert( all_fn.end(), pos_fn.begin(), pos_fn.end());
	all_fn.insert( all_fn.end(), neg_fn.begin(), neg_fn.begin() +pos_fn.size());

	//initializing the problem
	prob.l = all_fn.size();
	svm_node** x = new svm_node *[prob.l];

	//setting the y values
	prob.y = new double[prob.l];
	for(int i=0; i<pos_fn.size(); i++) prob.y[i]=1;
	for(int i=0; i<all_fn.size()-pos_fn.size(); i++) prob.y[i+pos_fn.size()]=-1;


	//Iterate through images to get HOG values
	Filenames::iterator it;
	int i = 0;
	std::cout << all_fn.size() <<std::endl;
	for(it=all_fn.begin() ; it < all_fn.end(); it++,i++) {
		//std::cout << "getting data: "<< i  << std::endl;
		cv::Mat temp_image= cv::imread(*it,1);
		FeatureVec features;
		cv::HOGDescriptor hogdis;
		hogdis.compute(temp_image, features);

		int num_l=0;
		GetSparseFeatLength(features, num_l);
		svm_node *x_space = new svm_node[num_l+1];
		MakeSparseFeatures (features, x_space);


		x[i] = x_space;
		temp_image.release();
		//delete x_space;
	}
	prob.x = x;


	std::cout << "end of training data" << std::endl;

}

void GetSparseFeatLength(FeatureVec &features, int &num_l){
	for (int k=0; k<features.size(); k++){
		if(features[k]>0){
			num_l++;
		}
	}
}

void MakeSparseFeatures (FeatureVec &features, svm_node *x_space){

	int j=0;
	for (int k=0; k<features.size(); k++){
		if(features[k]>0){
			//std::cout << "sparsing " <<k << std::endl;
			x_space[j].index = j+1;
			x_space[j].value = features[k];
			j++;
		}
	}

	x_space[j].index = -1;
}

void GetTestData(svm_node* testnode, const svm_model *model){
//	cv::Mat test_image;
//	fs::path test_path =fs::current_path(); test_path /= "/src/train_set/pos_rsz/crop_000001a0.png";
//	std::string im_path = fs::canonical(test_path).string();

	Filenames pos_fn, neg_fn;
	fs::path pos_path =fs::current_path(); pos_path /= "/src/test_set/pos_new";
	fs::path neg_path =fs::current_path(); neg_path /= "/src/test_set/neg_new";
	GetFileNames (pos_fn, pos_path);
	GetFileNames (neg_fn, neg_path);

	std::cout << pos_fn.size() << std::endl;
	std::cout << neg_fn.size() << std::endl;

	//Merge all filenames together into new vector
	Filenames all_fn;
	all_fn.reserve( pos_fn.size() + neg_fn.size()); // preallocate memory
	all_fn.insert( all_fn.end(), pos_fn.begin(), pos_fn.end());
	all_fn.insert( all_fn.end(), neg_fn.begin(), neg_fn.end());


	//Iterate through images to get HOG values
	Filenames::iterator it;
	int i = 0;
	std::cout << "all_fn" << all_fn.size() <<std::endl;
	for(it=all_fn.begin() ; it < all_fn.end(); it++,i++) {
		std::cout << "getting data: "<< *it  << std::endl;
		cv::Mat temp_image= cv::imread(*it,1);
		double scale = 128.0/temp_image.rows;
		std::cout << "scale : " << scale << std::endl;
		resize(temp_image, temp_image, cv::Size(), scale, scale, cv::INTER_LINEAR);


		FeatureVec features;
		cv::HOGDescriptor hogdis;
		hogdis.compute(temp_image, features);

		std::cout << "longer leg: " << temp_image.rows << " : " << temp_image.cols <<std::endl;
		int num_win = features.size()/3780;
		std::cout << "num windows: " << features.size()/3780 <<std::endl;
		std::vector<double> predictions;
		for (int j = 0; j<num_win; j++){
			int num_l=0;
			//std::vector<int> v1(v.begin() + 4, v.end() - 2);
			int start_val = (j)*3780; int end_val = (j+1)*3780;
			FeatureVec features_sub(features.begin()+start_val,features.begin()+end_val);
			GetSparseFeatLength(features_sub, num_l);
			svm_node *x_space = new svm_node[num_l+1];
			MakeSparseFeatures (features_sub, x_space);
			//std::cout << features_sub.size() << std::endl;

			double prob_est[2];
			predictions.push_back(svm_predict_probability(model, x_space, prob_est));
			if (predictions[j]==1){
				printf("%f\t%f\t%f\n", predictions[j], prob_est[0], prob_est[1]);
			}
		}

		//int init =0;
		//std::cout << std::accumulate(predictions.begin(),predictions.end(),0.0)/(features.size()*1.0/3780.0) <<std::endl;;

		cv::namedWindow( "Display window" );

		for(int k = 0; k<num_win; k++){
			if (predictions[k]==1){
				std::cout << "drawing rect :" << k <<std::endl;
				cv::Point pt1, pt2;
				pt1.y = 0; if (k>0) pt1.x = k*8-1; else pt1.x = k*8;
				pt2.x = pt1.x+63; pt2.y = pt1.y+127;
				std::cout << pt1 <<std::endl;
				std::cout << pt2 <<std::endl;
				cv::rectangle(temp_image, pt1, pt2, cv::Scalar(0, 255, 0), 2, CV_AA);
			}
		}

		cv::imshow( "Display window", temp_image );
		cv::waitKey(5000);
		temp_image.release();
	}



	std::cout << "end of testing data" << std::endl;

}

void TestData(svm_node* testnode){
//	cv::Mat test_image;
	fs::path test_path =fs::current_path(); test_path /= "/src/train_set/pos_rsz/crop_000001a0.png";
	std::string im_path = fs::canonical(test_path).string();


	//Iterate through images to get HOG values

	int i = 0;

	std::cout << "getting data: "<< im_path  << std::endl;
	cv::Mat temp_image= cv::imread(im_path,1);
	FeatureVec features;
	cv::HOGDescriptor hogdis;
	hogdis.compute(temp_image, features);

	int num_l=0;
	GetSparseFeatLength(features, num_l);
	svm_node *x_space = new svm_node[num_l+1];
	MakeSparseFeatures (features, x_space);

	std::cout << x_space[0].value <<std::endl;

	testnode = x_space;
	//free(x_space);
	temp_image.release();

	std::cout << "end of test data" << std::endl;

}

void GetFileNames (Filenames &fn, fs::path &directory){
	fs::directory_iterator end_iter;

	if ( fs::exists(directory) && fs::is_directory(directory)){
		for( fs::directory_iterator dir_iter(directory) ; dir_iter != end_iter ; ++dir_iter){
			if (fs::is_regular_file(dir_iter->status()) ){
				if(fs::is_regular_file(*dir_iter) && (dir_iter->path().extension() == ".png" || dir_iter->path().extension() == ".jpg")){

					fs::directory_entry& entry = (*dir_iter);
					std::string temp_path = fs::canonical(entry).string();
					fn.push_back(temp_path);

				}
			}
		}
	}
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

						fs::path newDir =fs::current_path(); newDir /= "src/train_set/pos_rsz/";
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
