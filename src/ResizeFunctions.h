/*
 * ResizeFunctions.h
 *
 *  Created on: Feb 11, 2015
 *      Author: agata
 */

#ifndef RESIZEFUNCTIONS_H_
#define RESIZEFUNCTIONS_H_





#endif /* RESIZEFUNCTIONS_H_ */

void ResizeImages(fs::path &directory);
void ResizeImagesRandom(fs::path &directory);
void ResizeToScale(cv::Mat &temp_image, float val);
void ResizeImagesRandomToScale(fs::path &directory);
