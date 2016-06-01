 

#include "depth_estimation.h"

int main()
{
    string dir_train_images = string("/home/gordon/TrainingData/data1/jpgonlydata/");
    string dir_train_labels = string("/home/gordon/TrainingData/data1/pngonlydata/");
    string dir_test_labels = string("/home/gordon/tiny-cnn-master/depthconstruction/testimages/labels");
    string dir_test_images = string("/home/gordon/tiny-cnn-master/depthconstruction/testimages/");
    DepthEstimation depth;
    depth.TrainingNetwork(dir_train_images, dir_train_labels, dir_test_labels, dir_test_images);
    return 0;
}