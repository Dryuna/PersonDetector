//============================================================================
// Name        : PersonDetector.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "PersonDetector.h"
#include "ResizeFunctions.h"


typedef std::vector<fs::path> Filepaths;


int main() {
	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
    fs::path someDir =fs::current_path(); someDir /= "src/neg";
    //ResizeImages(someDir);
	//ResizeImagesRandom(someDir);
	//ResizeImagesRandomToScale(someDir);

//	fs::path imgDir =fs::current_path(); imgDir /= "src/ws2.png";
//	std::string im_loc = fs::canonical(imgDir).string();
//	cv::Mat image= cv::imread(im_loc, CV_LOAD_IMAGE_COLOR);
//	if( !image.data )  { return -1; }

	//cv::Mat im_grey;
	//cv::cvtColor(image, im_grey, CV_BGR2GRAY);

	//GetTrainingData();

	int test = TestLibSVM();


	return 0;
}


int TestLibSVM(){

	Filepaths fp;
	fs::path pos_path =fs::current_path(); pos_path /= "/src/train_set/pos_rsz";
	fs::path neg_path =fs::current_path(); neg_path /= "/src/train_set/neg_rsz2";
	fp.push_back(pos_path); fp.push_back(neg_path);

	svm_model *model;
	int train_now =0;
	if (train_now ==1){
		std::cout << "starting svm_training" << std::endl;
		svm_parameter param;
		GetTrainingParameters(param);
		svm_problem prob;


		for(int i = 0; i<1; i++){
			std::cout << "time is : " << i << std::endl;
			GetTrainingData(prob, -1, fp);
			int nr_fold=10;
			double *target = new double[prob.l];
			svm_cross_validation(&prob, &param, nr_fold, target);
		}

		GetTrainingData(prob, -1, fp);
		model= svm_train(&prob,&param);
	    free(prob.y);
	    free(prob.x);
		const char *model_file_name = "saved_model.model";
		int saved= svm_save_model(model_file_name, model);
	    svm_destroy_param(&param);

	}

	else{
		const char *model_file_name = "saved_model.model";
		model =svm_load_model(model_file_name);
	}


	std::cout << "testing training" << std::endl;
	svm_node* testnode;
	//TestTrainingData(testnode, model, fp);

	Filepaths fp2;
	fs::path pos_path2 =fs::current_path(); pos_path2 /= "/src/test_set/pos_new";
	fs::path neg_path2 =fs::current_path(); neg_path2 /= "/src/test_set/neg_new";
	fp2.push_back(pos_path2); fp2.push_back(neg_path2);
	svm_node* testnode2;
	TestTrainingData(testnode2, model, fp2);

	std::cout << "end of demo" << std::endl;

	return 0;

}

void GetTrainingParameters(svm_parameter &param){
	//Parameters
	//svm_parameter param;
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
}

void GetTrainingData(svm_problem &prob, int size_set_in, Filepaths &fp){

	Filenames pos_fn, neg_fn;
	GetFileNames (pos_fn, fp[0]);
	GetFileNames (neg_fn, fp[1]);

	std::cout << pos_fn.size() << std::endl;
	std::cout << neg_fn.size() << std::endl;

	//Merge all filenames together into new vector
	Filenames all_fn;
	int size_set;
	if (size_set_in==2) size_set = pos_fn.size();
	else size_set=-1;
	// = pos_fn.size();
	GetTrainSet (pos_fn, neg_fn, all_fn, size_set);
	std::cout << "numtrain is : " << all_fn.size() <<std::endl;
//	all_fn.reserve( pos_fn.size() + neg_fn.size() ); // preallocate memory
//	all_fn.insert ( all_fn.end(), pos_fn.begin(), pos_fn.end());
//	all_fn.insert ( all_fn.end(), neg_fn.begin(), neg_fn.begin() +pos_fn.size());

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

void GetTrainSet (Filenames &pos_fn, Filenames &neg_fn, Filenames &all_fn, int size_set){
	std::cout << "getting train set" <<std::endl;
	int tot_sz;
	if (size_set==-1) {
			tot_sz = pos_fn.size() + neg_fn.size();
			all_fn.reserve( tot_sz ); // preallocate memory
			all_fn.insert ( all_fn.end(), pos_fn.begin(), pos_fn.end());
			all_fn.insert ( all_fn.end(), neg_fn.begin(), neg_fn.end());
	}
		//tot_sz=pos_fn.size()+neg_fn.size();
	else {
		tot_sz=2*size_set; // preallocate memory
		//std::cout << pos_fn.size()+neg_fn.size() <<std::endl;
		all_fn.reserve( tot_sz );
		  // set some values:
		std::vector<int> rand_idx_pos, rand_idx_neg;
		for (int i=0; i<pos_fn.size(); ++i) rand_idx_pos.push_back(i);
		for (int i=0; i<neg_fn.size(); ++i) rand_idx_neg.push_back(i);

		// using built-in random generator:
		std::random_shuffle ( rand_idx_pos.begin(), rand_idx_pos.end() );
		std::random_shuffle ( rand_idx_neg.begin(), rand_idx_neg.end() );

		std::vector<int>::iterator it_pos, it_neg;
		for(it_pos=rand_idx_pos.begin(); it_pos<rand_idx_pos.end(); ++it_pos){
			int temp_idx = rand_idx_pos.back();
			all_fn.push_back (pos_fn[temp_idx]);
			rand_idx_pos.pop_back();
			//std::cout << *it_pos <<std::endl;
		}

		int add_neg = tot_sz-rand_idx_pos.size();
		for(it_neg=rand_idx_neg.begin(); it_neg<rand_idx_neg.begin()+add_neg; ++it_neg){
			int temp_idx = rand_idx_neg.back();
			all_fn.push_back (neg_fn[temp_idx]);
			rand_idx_neg.pop_back();
		}
	}

	std::cout << "leaving train set" << size_set<< ": " << all_fn.size() <<std::endl;
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


void TestTrainingData(svm_node* testnode, const svm_model *model, Filepaths &fp){
//	cv::Mat test_image;
//	fs::path test_path =fs::current_path(); test_path /= "/src/train_set/pos_rsz/crop_000001a0.png";
//	std::string im_path = fs::canonical(test_path).string();

	Filenames pos_fn, neg_fn;
	GetFileNames (pos_fn, fp[0]);
	GetFileNames (neg_fn, fp[1]);

	std::cout << pos_fn.size() << std::endl;
	std::cout << neg_fn.size() << std::endl;

	std::vector<int>test_labels, pred_labels;
	for (int i = 0; i<pos_fn.size(); i++) test_labels.push_back(1);
	for (int i = 0; i<neg_fn.size(); i++) test_labels.push_back(-1);

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
		int pred=0;
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
				pred=1;
			}
		}


		if(pred==1){
			pred_labels.push_back(1);
		}
		else{
			pred_labels.push_back(-1);
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
		cv::waitKey(1500);
		temp_image.release();
	}

	int total_correct=0;
	for(int i = 0; i<pred_labels.size(); i++){
		if (pred_labels[i] == test_labels[i]){
			total_correct++;
		}
	}

	float perc_right = ((float)total_correct)/pred_labels.size();

	std::cout << "end of testing data: " << perc_right << std::endl;

}


void GetFileNames (Filenames &fn, fs::path &directory){
	fs::directory_iterator end_iter;

	if ( fs::exists(directory) && fs::is_directory(directory)){
		for( fs::directory_iterator dir_iter(directory) ; dir_iter != end_iter ; ++dir_iter){
			if (fs::is_regular_file(dir_iter->status()) ){
				if(fs::is_regular_file(*dir_iter) && (dir_iter->path().extension() == ".png" || dir_iter->path().extension() == ".jpg" || dir_iter->path().extension() == ".JPEG")){

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





