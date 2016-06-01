#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include </home/gordon/tiny-cnn-master/tiny_cnn/tiny_cnn.h>
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
using namespace std;

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;
/*function... might want it in some class?*/
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}
void convertimages(const string& imagefilename,double scale,
                   int w,
                   int h,vector<vec_t>& data){
    //cout << imagefilename << endl;
    Mat img = imread(imagefilename,IMREAD_GRAYSCALE);
    if (img.data == nullptr) return;
    Mat_<uint8_t> resized;
    resize(img, resized,Size(w, h));
    vec_t d;
    transform(resized.begin(), resized.end(), back_inserter(d),
                   [=](uint8_t c) { return c * scale; });
    data.push_back(d);
    
}



void sample_mlp(void) {
    using namespace tiny_cnn;
    network<mse, gradient_descent> nn;

    nn << fully_connected_layer<sigmoid>(320 * 320, 300)
       << fully_connected_layer<identity>(300, 48*48);

    assert(nn.in_dim() == 320 * 320);
    assert(nn.out_dim() == 48*48);
       
    cout << "load models..." << endl;
    
    string dirtrainimages = string("/home/gordon/TrainingData/data1/jpgonlydata/");
    string dirtrainlabels = string("/home/gordon/TrainingData/data1/pngonlydata/");
    string dirtestlabels = string("/home/gordon/tiny-cnn-master/depthconstruction/testimages/labels");
    string dirtestimages = string("/home/gordon/tiny-cnn-master/depthconstruction/testimages/");
    
    vector<string> files = vector<string>();
    getdir(dirtrainimages,files);
    vector<vec_t> trainimages,testimages;
    vector<vec_t> trainlabels,testlabels;
    
    
    for (unsigned int i = 0;i < files.size();i++) {
        converimages(files[i],0.00392157,320,320,trainimages);
    }
    
    vector<string> files1 = vector<string>();
    getdir(dirtrainlabels,files1);
    
    for (unsigned int i = 0;i < files1.size();i++) {
        converimages(files1[i],0.00392157,48,48,trainlabels);
    }
    vector<string> files2 = vector<string>();
    getdir(dirtestimages,files2);
    
    for (unsigned int i = 0;i < files2.size();i++) {
        converimages(files2[i],0.00392157,320,320,testimages);
    }
    
    vector<string> files3 = vector<string>();
    getdir(dirtestlabels,files3);
    
    for (unsigned int i = 0;i < files3.size();i++) {
        converimages(files3[i],0.00392157,48,48,testlabels);
    }

    boost::progress_display disp(trainimages.size());
    boost::timer t;
    
    
    int minibatch_size = 10;
    int num_epochs = 30;
    
    nn.optimizer().alpha = 0.001;
    
    auto on_enumerate_epoch = [&](){
        cout << t.elapsed() << "s elapsed." << endl;
        
        auto res = nn.test(testimages);
        
        cout << nn.optimizer().alpha << endl;
        nn.optimizer().alpha *= 0.95;
        nn.optimizer().alpha = max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);
        disp.restart(trainimages.size());
        t.restart();
    };
   
    auto on_enumerate_data = [&](){ 
        ++disp; 
    };
    cout << "start learning" << endl;
    nn.train(trainimages, trainlabels, minibatch_size, num_epochs, on_enumerate_data, on_enumerate_epoch);
    cout << "end training." << endl;

    

    ofstream ofs("Deepnet-weights");
    ofs << nn;
}

int main()
{
    
    sample_mlp();

    
}