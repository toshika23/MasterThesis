
#include "depth_estimation.h"




int DepthEstimation::GetDir (string source_dir, vector<string> &input_files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(source_dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << source_dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        input_files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    
    return 0;
}


void DepthEstimation::ConvertImagesToVect (string directory,const string& image_filename,double scale_parameter,
                   int image_width,
                   int image_height,vector<vec_t>& output_vec_t)
{
    //cout<<directory<<endl;
    string image_filename1 = directory + image_filename;
    //string image_filename1 =  image_filename;
    Mat input_img = imread(image_filename1,IMREAD_GRAYSCALE);
    //cout<<"input_img.data"<<input_img<<endl;
    if (  input_img.data == nullptr  ) return;
    
    Mat_<uint8_t> resized;
    resize(input_img, resized,Size(image_width, image_height));
    vec_t pixel_value;
    transform(resized.begin(), resized.end(), back_inserter(pixel_value),
                   [=](uint8_t c) { return c * scale_parameter; });
    
    output_vec_t.push_back(pixel_value);
    
}



void DepthEstimation::TrainingNetwork(string dir_train_images, string dir_train_labels, string dir_test_labels, string dir_test_images)
{
    try {
    network<mse, gradient_descent> nn;
    /*nn << fully_connected_layer<sigmoid>(640 * 480,120000)
       << convolutional_layer<tan_h>(400,300,12,1,12)
       << max_pooling_layer<sigmoid>(400,300,12,4,2)
       << convolutional_layer<tan_h>(240,180, 8, 1, 10)
       << average_pooling_layer<tan_h>(120, 80, 10, 4)
       << fully_connected_layer<sigmoid>(64 * 48 * 10, 3072);*/
    /* first try 
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 180x90in, conv5x5, 1-6 f-maps
        << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28in, 6 f-maps, pool2x2
        << fully_connected_layer<tan_h>(14 * 14 * 6, 120)
        << fully_connected_layer<identity>(120, 1024);*/
    
    nn << convolutional_layer<relu>(320, 320, 11, 1, 96) // 180x90in, conv5x5, 1-6 f-maps
        << max_pooling_layer<sigmoid>(310, 310, 96, 5) // 28x28in, 6 f-maps, pool2x2
        << convolutional_layer<relu>(62,62,6,96,64)
        << max_pooling_layer<sigmoid>(57, 57, 64,4,2)
        << fully_connected_layer<identity>(50176, 48*48);
        
        
    assert(nn.in_dim() == 320 * 320);
    assert(nn.out_dim() == 48*48); 
    
    vector<vec_t> train_images,test_images;
    vector<vec_t> train_labels,test_labels;
    
    vector<string> train_images_vector = vector<string>();
    GetDir(dir_train_images,train_images_vector);
    
    for (unsigned int i = 0;i < train_images_vector.size();i++) {
        ConvertImagesToVect(dir_train_images,train_images_vector[i],0.00392157,320,320,train_images);
    }
    
    vector<string> train_labels_vector = vector<string>();
    GetDir(dir_train_labels,train_labels_vector);
    
    for (unsigned int i = 0;i < train_labels_vector.size();i++) {
        ConvertImagesToVect(dir_train_labels,train_labels_vector[i],0.00392157,48,48,train_labels);
    }
    vector<string> test_images_vector = vector<string>();
    GetDir(dir_test_images,test_images_vector);
    
    for (unsigned int i = 0;i < test_images_vector.size();i++) {
        ConvertImagesToVect(dir_test_images,test_images_vector[i],0.00392157,320,320,test_images);
    }
    
    vector<string> test_labels_vector = vector<string>();
    GetDir(dir_test_labels,test_labels_vector);
    
    for (unsigned int i = 0;i < test_labels_vector.size();i++) {
        ConvertImagesToVect(dir_test_labels,test_labels_vector[i],0.00392157,48,48,test_labels);
    }
    
    boost::progress_display disp(train_images.size());
    //cout<<"size"<<train_images.size()<<endl;
    boost::timer t;
    
    
    int minibatch_size = 10;
    int num_epochs = 30;
    
    nn.optimizer().alpha = 0.001;
    int processed=0;
    auto on_enumerate_epoch = [&](){
        
        static unsigned int call_count = 0;
        
        
        cout << "Time taken to train----"<< t.elapsed()  << endl;
        
        float time_elapsed=t.elapsed();
        float total_time=num_epochs*time_elapsed;
        float remaining_time_ms=total_time-time_elapsed;
        float remaining_time_sec=(remaining_time_ms/1000);
        float remaining_time_min=remaining_time_sec/60;
        float remaining_time_hour=remaining_time_min/60;
        float processed_percentage=(time_elapsed/total_time)*100;
        float remaining_percentage=100-processed_percentage;
        
        //cout << "Total traning time----" << total_time <<""<< "milliseconds"<< endl;
        //cout << t.elapsed() << "Time taken to train" << endl;
        //cout << "Total traning finished in percentage ----" <<""<< processed_percentage << "%"<< endl;
        //cout << "Remaining percentage ----" << remaining_percentage <<""<< "%"<< endl;
        //cout << "Remaining time required ----" << remaining_time_hour <<""<< ":"<<""<< remaining_time_min <<""<< "HH:MM"<< endl;
        
        auto res = nn.test(test_images);
        cout << nn.optimizer().alpha << endl;
        nn.optimizer().alpha *= 0.95;
        nn.optimizer().alpha = max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);
        
        disp.restart(train_images.size());
        t.restart();
        call_count++;
        
        if (call_count%num_epochs==0)
        {
            int W=320;
            int H=320;       
            //cout<<"ohhhooo"<<res.size()<<endl;
            for (int i = 0; i < res.size(); i++){
            Mat_<uint8_t> out(H,W);
            int k=0;
            for(int y=0;  y<H; y++)
                for (int x=0;  x<W; x++)
                    out.data[y*W+x]=res[i][k++]*255.0;
            const char *path1 = "/home/gordon/tiny-cnn-master/depthconstruction/totest/" ;
            char numstr[30];
            sprintf(numstr, "%d", i);
            string result = path1 + string (numstr) + ".png";
            imwrite(result,out);
            }
        }
        if (call_count%10==0)
        {
            /*stringstream file_name;
            file_name << flags.model_file_ << call_count;
            result = file_name.str();
            ofstream ofs("result");
            ofs << nn;*/
            const char *path1 = "/home/gordon/tiny-cnn-master/depthconstruction/totest/" ;
            ofstream llh_file;
            ofstream myfile;
            string result;
            char numstr[30];
            sprintf(numstr, "%d", call_count);
            result = path1 + string (numstr) + ".txt";
            myfile.open(result.c_str());
            myfile << nn;
        }
        cout <<call_count<<endl;
        
    };
   
    auto on_enumerate_data = [&](){ 
        ++disp; 
    };
    cout << "start learning" << endl;
    nn.train(train_images, train_labels, minibatch_size, num_epochs, on_enumerate_data, on_enumerate_epoch);
    cout << "end training." << endl;
    ofstream ofs("Deepnet-weights");
    ofs << nn;
    } catch (const nn_error& e) {
   cout << e.what();
    }
    
}

