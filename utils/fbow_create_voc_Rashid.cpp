
//First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
// #include <fstream>
#include <vector>
#include "fbow.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#ifdef USE_CONTRIB
    #include <opencv2/xfeatures2d/nonfree.hpp>
    #include <opencv2/xfeatures2d.hpp>
#endif

#include "dirreader.h"


//Second step,creates the vocabulary from the set of features. It can be slow
#include "vocabulary_creator.h"


using namespace fbow;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait(){
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

vector< cv::Mat > loadFeatures( std::vector<string> path_to_images, string descriptor = "")  {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")        fdetector=cv::ORB::create(2000);
    else if (descriptor == "brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor == "akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor == "surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector< cv::Mat > features;


    cout << "Extracting features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty()) {
            std::cerr << "Could not open image:" << path_to_images[i] << std::endl;
            continue;
        }
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        cout << "extracting features: total= " << keypoints.size() << endl;
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv){
    try{
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc<4){
            cerr << "Usage:  descriptor_name output.fbow dir_with_images \n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)" << endl;
            return -1;
        }

        string descriptor = argv[1];
        string output = argv[2];

        auto images = DirReader::read( argv[3]);
        vector< cv::Mat > features = loadFeatures(images, descriptor);

        cout << "DescName = " << descriptor << endl;
        fbow::VocabularyCreator::Params params;
        params.k = stoi(cml("-k","10"));
        params.L = stoi(cml("-l","6"));
        params.nthreads = stoi(cml("-t","1"));
        params.maxIters = std::stoi (cml("-maxIters","0"));
        params.verbose = cml["-v"];
        srand(0);
        fbow::VocabularyCreator voc_creator;
        fbow::Vocabulary voc;
        cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        voc_creator.create(voc, features, descriptor, params);
        auto t_end = std::chrono::high_resolution_clock::now();
        cout << "time = " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count()) << " msecs" << endl;
        cout << "nblocks = "<< voc.size() << endl;
        cerr << "Saving " << output << endl;
        voc.saveToFile(output);


    }catch(std::exception &ex){
        cerr << ex.what() << endl;
    }

    return 0;
}
