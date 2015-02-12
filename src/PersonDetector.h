/*
 * PersonDetector.h
 *
 *  Created on: Feb 7, 2015
 *      Author: agata
 */

#ifndef PERSONDETECTOR_H_
#define PERSONDETECTOR_H_
#endif /* PERSONDETECTOR_H_ */


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
typedef std::vector<fs::path> Filepaths;

int TestLibSVM();
void GetFileNames (Filenames &fn, fs::path &directory);
void GetTrainingData(svm_problem &prob, int size_set_in, Filepaths &fp);
void MakeSparseFeatures (FeatureVec &features, svm_node *x_space);
void GetSparseFeatLength(FeatureVec &features, int &num_l);
//void TestData(svm_node* testnode);
void GetTrainSet (Filenames &pos_fn, Filenames &neg_fn, Filenames &all_fn, int size_set);
void GetTrainingParameters(svm_parameter &param);
void TestTrainingData(svm_node* testnode, const svm_model *model, Filepaths &fp);
void DrawGrid(cv::Mat &img_out, int dist, cv::Scalar color, cv::Point offset);
