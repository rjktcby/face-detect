#import <ShapeRegressor.h>

#ifdef __cplusplus

#include <iostream>
#include <cstdio>
#include <cstdlib>
// #include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>
#include <utility>

#endif // __cplusplus

class BoundingBox
{
    public:
        double start_x;
        double start_y;
        double width;
        double height;
        double centroid_x;
        double centroid_y;
        BoundingBox(){
            start_x = 0;
            start_y = 0;
            width = 0;
            height = 0;
            centroid_x = 0;
            centroid_y = 0;
        }; 
};


class Fern
{
private:
    int fern_pixel_num_;
    int landmark_num_;
    cv::Mat_<int> selected_nearest_landmark_index_;
    cv::Mat_<double> threshold_;
    cv::Mat_<int> selected_pixel_index_;
    cv::Mat_<double> selected_pixel_locations_;
    std::vector<cv::Mat_<double> > bin_output_;
public:
    std::vector<cv::Mat_<double> > Train(const std::vector<std::vector<double> >& candidate_pixel_intensity, 
                                         const cv::Mat_<double>& covariance,
                                         const cv::Mat_<double>& candidate_pixel_locations,
                                         const cv::Mat_<int>& nearest_landmark_index,
                                         const std::vector<cv::Mat_<double> >& regression_targets,
                                         int fern_pixel_num);
    cv::Mat_<double> Predict(const cv::Mat_<uchar>& image,
                             const cv::Mat_<double>& shape,
                             const cv::Mat_<double>& rotation,
                             const BoundingBox& bounding_box,
                             double scale);
    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
};

class FernCascade
{
public:
    std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
                                         const std::vector<cv::Mat_<double> >& current_shapes,
                                         const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                                         const std::vector<BoundingBox> & bounding_box,
                                         const cv::Mat_<double>& mean_shape,
                                         int second_level_num,
                                         int candidate_pixel_num,
                                         int fern_pixel_num,
                                         int curr_level_num,
                                         int first_level_num);  
    cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, 
                             const BoundingBox& bounding_box, 
                             const cv::Mat_<double>& mean_shape,
                             const cv::Mat_<double>& shape);
    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
private:
    std::vector<Fern> ferns_;
    int second_level_num_;
};

class ShapeRegressor
{
public:
    ShapeRegressor(); 
    void Train(const std::vector<cv::Mat_<uchar> >& images, 
               const std::vector<cv::Mat_<double> >& ground_truth_shapes,
               const std::vector<BoundingBox>& bounding_box,
               int first_level_num, int second_level_num,
               int candidate_pixel_num, int fern_pixel_num,
               int initial_num);
    cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
    void Load(std::string path);
    void Save(std::string path);
private:
    int first_level_num_;
    int landmark_num_;
    std::vector<FernCascade> fern_cascades_;
    cv::Mat_<double> mean_shape_;
    std::vector<cv::Mat_<double> > training_shapes_;
    std::vector<BoundingBox> bounding_box_;
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2, 
                         cv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const std::vector<double>& v_1, 
                            const std::vector<double>& v_2);

#ifdef __cplusplus
using namespace std;
using namespace cv;
#endif // __cplusplus

ShapeRegressor::ShapeRegressor(){
    first_level_num_ = 0;
}

/**
 * @param images gray scale images
 * @param ground_truth_shapes a vector of N*2 matrix, where N is the number of landmarks
 * @param bounding_box BoundingBox of faces
 * @param first_level_num number of first level regressors
 * @param second_level_num number of second level regressors
 * @param candidate_pixel_num number of pixels to be selected as features
 * @param fern_pixel_num number of pixel pairs in a fern
 * @param initial_num number of initial shapes for each input image
 */
void ShapeRegressor::Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num){
    cout<<"Start training..."<<endl;
    bounding_box_ = bounding_box;
    training_shapes_ = ground_truth_shapes;
    first_level_num_ = first_level_num;
    landmark_num_ = ground_truth_shapes[0].rows; 
    // data augmentation and multiple initialization 
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;
     
    RNG random_generator(getTickCount());
    for(int i = 0;i < images.size();i++){
        for(int j = 0;j < initial_num;j++){
            int index = 0;
            do{
                // index = (i+j+1) % (images.size()); 
                index = random_generator.uniform(0, images.size());
            }while(index == i);
            augmented_images.push_back(images[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_box.push_back(bounding_box[i]); 
            // 1. Select ground truth shapes of other images as initial shapes
            // 2. Project current shape to bounding box of ground truth shapes 
            Mat_<double> temp = ground_truth_shapes[index];
            temp = ProjectShape(temp, bounding_box[index]);
            temp = ReProjectShape(temp, bounding_box[i]);
            current_shapes.push_back(temp); 
        } 
    }
    
    // get mean shape from training shapes
    mean_shape_ = GetMeanShape(ground_truth_shapes,bounding_box); 
    
    // train fern cascades
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0;i < first_level_num;i++){
        cout<<"Training fern cascades: "<<i+1<<" out of "<<first_level_num<<endl;
        prediction = fern_cascades_[i].Train(augmented_images,current_shapes,
                augmented_ground_truth_shapes,augmented_bounding_box,mean_shape_,second_level_num,candidate_pixel_num,fern_pixel_num, i+1, first_level_num);
        
        // update current shapes 
        for(int j = 0;j < prediction.size();j++){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j],augmented_bounding_box[j]);
        }
    } 
    
}


void ShapeRegressor::Write(ofstream& fout){
    fout<<first_level_num_<<endl;
    fout<<mean_shape_.rows<<endl;
    for(int i = 0;i < landmark_num_;i++){
        fout<<mean_shape_(i,0)<<" "<<mean_shape_(i,1)<<" "; 
    }
    fout<<endl;
    
    fout<<training_shapes_.size()<<endl;
    for(int i = 0;i < training_shapes_.size();i++){
        fout<<bounding_box_[i].start_x<<" "<<bounding_box_[i].start_y<<" "
            <<bounding_box_[i].width<<" "<<bounding_box_[i].height<<" "
            <<bounding_box_[i].centroid_x<<" "<<bounding_box_[i].centroid_y<<endl;
        for(int j = 0;j < training_shapes_[i].rows;j++){
            fout<<training_shapes_[i](j,0)<<" "<<training_shapes_[i](j,1)<<" "; 
        }
        fout<<endl;
    }
    
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].Write(fout);
    } 
}

void ShapeRegressor::Read(ifstream& fin){
    fin>>first_level_num_;
    fin>>landmark_num_;
    mean_shape_ = Mat::zeros(landmark_num_,2,CV_64FC1);
    for(int i = 0;i < landmark_num_;i++){
        fin>>mean_shape_(i,0)>>mean_shape_(i,1);
    }
    
    int training_num;
    fin>>training_num;
    training_shapes_.resize(training_num);
    bounding_box_.resize(training_num);

    for(int i = 0;i < training_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height>>temp.centroid_x>>temp.centroid_y;
        bounding_box_[i] = temp;
        
        Mat_<double> temp1(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>temp1(j,0)>>temp1(j,1);
        }
        training_shapes_[i] = temp1; 
    }

    fern_cascades_.resize(first_level_num_);
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].Read(fin);
    }
} 


Mat_<double> ShapeRegressor::Predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num){
    // generate multiple initializations
    Mat_<double> result = Mat::zeros(landmark_num_,2, CV_64FC1);
    RNG random_generator(getTickCount());
    for(int i = 0;i < initial_num;i++){
        random_generator = RNG(i);
        int index = random_generator.uniform(0,training_shapes_.size());
        Mat_<double> current_shape = training_shapes_[index];
        BoundingBox current_bounding_box = bounding_box_[index];
        current_shape = ProjectShape(current_shape,current_bounding_box);
        current_shape = ReProjectShape(current_shape,bounding_box);
        for(int j = 0;j < first_level_num_;j++){
            Mat_<double> prediction = fern_cascades_[j].Predict(image,bounding_box,mean_shape_,current_shape);
            // update current shape
            current_shape = prediction + ProjectShape(current_shape,bounding_box);
            current_shape = ReProjectShape(current_shape,bounding_box); 
        }
        result = result + current_shape; 
    }    

    return 1.0 / initial_num * result;
}

void ShapeRegressor::Load(string path){
    cout<<"Loading model..."<<endl;
    ifstream fin;
    fin.open(path.c_str());
    if (!fin.good()) {
        throw std::string("Unable to open model from " + path);
    }
    this->Read(fin); 
    fin.close();
    cout<<"Model loaded successfully..."<<endl;
}

void ShapeRegressor::Save(string path){
    cout<<"Saving model..."<<endl;
    ofstream fout;
    fout.open(path.c_str());
    this->Write(fout);
    fout.close();
}

vector<Mat_<double> > Fern::Train(const vector<vector<double> >& candidate_pixel_intensity, 
                                  const Mat_<double>& covariance,
                                  const Mat_<double>& candidate_pixel_locations,
                                  const Mat_<int>& nearest_landmark_index,
                                  const vector<Mat_<double> >& regression_targets,
                                  int fern_pixel_num){
    // selected_pixel_index_: fern_pixel_num*2 matrix, the index of selected pixels pairs in fern
    // selected_pixel_locations_: fern_pixel_num*4 matrix, the locations of selected pixel pairs
    //                            stored in the format (x_1,y_1,x_2,y_2) for each row 
    fern_pixel_num_ = fern_pixel_num;
    landmark_num_ = regression_targets[0].rows;
    selected_pixel_index_.create(fern_pixel_num,2);
    selected_pixel_locations_.create(fern_pixel_num,4);
    selected_nearest_landmark_index_.create(fern_pixel_num,2);
    int candidate_pixel_num = candidate_pixel_locations.rows;

    // select pixel pairs from candidate pixels, this selection is based on the correlation between pixel 
    // densities and regression targets
    // for details, please refer to "Face Alignment by Explicit Shape Regression" 
    // threshold_: thresholds for each pair of pixels in fern 
    
    threshold_.create(fern_pixel_num,1);
    // get a random direction
    RNG random_generator(getTickCount());
    for(int i = 0;i < fern_pixel_num;i++){
        // RNG random_generator(i);
        Mat_<double> random_direction(landmark_num_ ,2);
        random_generator.fill(random_direction,RNG::UNIFORM,-1.1,1.1);

        normalize(random_direction,random_direction);
        vector<double> projection_result(regression_targets.size(), 0); 
        // project regression targets along the random direction 
        for(int j = 0;j < regression_targets.size();j++){
            double temp = 0;
            temp = sum(regression_targets[j].mul(random_direction))[0]; 
            projection_result[j] = temp;
        } 
        Mat_<double> covariance_projection_density(candidate_pixel_num,1);
        for(int j = 0;j < candidate_pixel_num;j++){
            covariance_projection_density(j) = calculate_covariance(projection_result,candidate_pixel_intensity[j]);
        }


        // find max correlation
        double max_correlation = -1;
        int max_pixel_index_1 = 0;
        int max_pixel_index_2 = 0;
        for(int j = 0;j < candidate_pixel_num;j++){
            for(int k = 0;k < candidate_pixel_num;k++){
                double temp1 = covariance(j,j) + covariance(k,k) - 2*covariance(j,k);
                if(abs(temp1) < 1e-10){
                    continue;
                }
                bool flag = false;
                for(int p = 0;p < i;p++){
                    if(j == selected_pixel_index_(p,0) && k == selected_pixel_index_(p,1)){
                        flag = true;
                        break; 
                    }else if(j == selected_pixel_index_(p,1) && k == selected_pixel_index_(p,0)){
                        flag = true;
                        break;
                    } 
                }
                if(flag){
                    continue;
                } 
                double temp = (covariance_projection_density(j) - covariance_projection_density(k))
                    / sqrt(temp1);
                if(abs(temp) > max_correlation){
                    max_correlation = temp;
                    max_pixel_index_1 = j;
                    max_pixel_index_2 = k;
                }
            }
        }

        selected_pixel_index_(i,0) = max_pixel_index_1; 
        selected_pixel_index_(i,1) = max_pixel_index_2; 
        selected_pixel_locations_(i,0) = candidate_pixel_locations(max_pixel_index_1,0);
        selected_pixel_locations_(i,1) = candidate_pixel_locations(max_pixel_index_1,1);
        selected_pixel_locations_(i,2) = candidate_pixel_locations(max_pixel_index_2,0);
        selected_pixel_locations_(i,3) = candidate_pixel_locations(max_pixel_index_2,1);
        selected_nearest_landmark_index_(i,0) = nearest_landmark_index(max_pixel_index_1); 
        selected_nearest_landmark_index_(i,1) = nearest_landmark_index(max_pixel_index_2); 

        // get threshold for this pair
        double max_diff = -1;
        for(int j = 0;j < candidate_pixel_intensity[max_pixel_index_1].size();j++){
            double temp = candidate_pixel_intensity[max_pixel_index_1][j] - candidate_pixel_intensity[max_pixel_index_2][j];
            if(abs(temp) > max_diff){
                max_diff = abs(temp);
            }
        }

        threshold_(i) = random_generator.uniform(-0.2*max_diff,0.2*max_diff); 
    } 

    // determine the bins of each shape
    vector<vector<int> > shapes_in_bin;
    int bin_num = pow(2.0,fern_pixel_num);
    shapes_in_bin.resize(bin_num);
    for(int i = 0;i < regression_targets.size();i++){
        int index = 0;
        for(int j = 0;j < fern_pixel_num;j++){
            double density_1 = candidate_pixel_intensity[selected_pixel_index_(j,0)][i];
            double density_2 = candidate_pixel_intensity[selected_pixel_index_(j,1)][i];
            if(density_1 - density_2 >= threshold_(j)){
                index = index + pow(2.0,j);
            } 
        }
        shapes_in_bin[index].push_back(i);
    }
     
    // get bin output
    vector<Mat_<double> > prediction;
    prediction.resize(regression_targets.size());
    bin_output_.resize(bin_num);
    for(int i = 0;i < bin_num;i++){
        Mat_<double> temp = Mat::zeros(landmark_num_,2, CV_64FC1);
        int bin_size = shapes_in_bin[i].size();
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            temp = temp + regression_targets[index]; 
        }
        if(bin_size == 0){
            bin_output_[i] = temp;
            continue; 
        }
        temp = (1.0/((1.0+1000.0/bin_size) * bin_size)) * temp;
        bin_output_[i] = temp;
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            prediction[index] = temp;
        }
    }
    return prediction;
}


void Fern::Write(ofstream& fout){
    fout<<fern_pixel_num_<<endl;
    fout<<landmark_num_<<endl;
    for(int i = 0;i < fern_pixel_num_;i++){
        fout<<selected_pixel_locations_(i,0)<<" "<<selected_pixel_locations_(i,1)<<" "
            <<selected_pixel_locations_(i,2)<<" "<<selected_pixel_locations_(i,3)<<" "<<endl;
        fout<<selected_nearest_landmark_index_(i,0)<<endl;
        fout<<selected_nearest_landmark_index_(i,1)<<endl;
        fout<<threshold_(i)<<endl;
    }
    for(int i = 0;i < bin_output_.size();i++){
        for(int j = 0;j < bin_output_[i].rows;j++){
            fout<<bin_output_[i](j,0)<<" "<<bin_output_[i](j,1)<<" ";
        }
        fout<<endl;
    } 

}

void Fern::Read(ifstream& fin){
    fin>>fern_pixel_num_;
    fin>>landmark_num_;
    selected_nearest_landmark_index_.create(fern_pixel_num_,2);
    selected_pixel_locations_.create(fern_pixel_num_,4);
    threshold_.create(fern_pixel_num_,1);
    for(int i = 0;i < fern_pixel_num_;i++){
        fin>>selected_pixel_locations_(i,0)>>selected_pixel_locations_(i,1)
            >>selected_pixel_locations_(i,2)>>selected_pixel_locations_(i,3);
        fin>>selected_nearest_landmark_index_(i,0)>>selected_nearest_landmark_index_(i,1);
        fin>>threshold_(i);
    }       
     
    int bin_num = pow(2.0,fern_pixel_num_);
    for(int i = 0;i < bin_num;i++){
        Mat_<double> temp(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>temp(j,0)>>temp(j,1);
        }
        bin_output_.push_back(temp);
    }
}

Mat_<double> Fern::Predict(const Mat_<uchar>& image,
                     const Mat_<double>& shape,
                     const Mat_<double>& rotation,
                     const BoundingBox& bounding_box,
                     double scale){
    int index = 0;
    for(int i = 0;i < fern_pixel_num_;i++){
        int nearest_landmark_index_1 = selected_nearest_landmark_index_(i,0);
        int nearest_landmark_index_2 = selected_nearest_landmark_index_(i,1);
        double x = selected_pixel_locations_(i,0);
        double y = selected_pixel_locations_(i,1);
        double project_x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width/2.0 + shape(nearest_landmark_index_1,0);
        double project_y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height/2.0 + shape(nearest_landmark_index_1,1);

        project_x = std::max(0.0,std::min((double)project_x,image.cols-1.0));
        project_y = std::max(0.0,std::min((double)project_y,image.rows-1.0)); 
        double intensity_1 = (int)(image((int)project_y,(int)project_x));

        x = selected_pixel_locations_(i,2);
        y = selected_pixel_locations_(i,3);
        project_x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width/2.0 + shape(nearest_landmark_index_2,0);
        project_y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height/2.0 + shape(nearest_landmark_index_2,1);
        project_x = std::max(0.0,std::min((double)project_x,image.cols-1.0));
        project_y = std::max(0.0,std::min((double)project_y,image.rows-1.0));
        double intensity_2 = (int)(image((int)project_y,(int)project_x));

        if(intensity_1 - intensity_2 >= threshold_(i)){
            index = index + (int)(pow(2,i));
        }
    }
    return bin_output_[index];
   
}

vector<Mat_<double> > FernCascade::Train(const vector<Mat_<uchar> >& images,
                                    const vector<Mat_<double> >& current_shapes,
                                    const vector<Mat_<double> >& ground_truth_shapes,
                                    const vector<BoundingBox> & bounding_box,
                                    const Mat_<double>& mean_shape,
                                    int second_level_num,
                                    int candidate_pixel_num,
                                    int fern_pixel_num,
                                    int curr_level_num, 
                                    int first_level_num){
    Mat_<double> candidate_pixel_locations(candidate_pixel_num,2);
    Mat_<int> nearest_landmark_index(candidate_pixel_num,1);
    vector<Mat_<double> > regression_targets;
    RNG random_generator(getTickCount());
    second_level_num_ = second_level_num;
    
    // calculate regression targets: the difference between ground truth shapes and current shapes
    // candidate_pixel_locations: the locations of candidate pixels, indexed relative to its nearest landmark on mean shape 
    regression_targets.resize(current_shapes.size()); 
    for(int i = 0;i < current_shapes.size();i++){
        regression_targets[i] = ProjectShape(ground_truth_shapes[i],bounding_box[i]) 
                                - ProjectShape(current_shapes[i],bounding_box[i]);
        Mat_<double> rotation;
        double scale;
        SimilarityTransform(mean_shape,ProjectShape(current_shapes[i],bounding_box[i]),rotation,scale);
        transpose(rotation,rotation);
        regression_targets[i] = scale * regression_targets[i] * rotation;
    }

    
    // get candidate pixel locations, please refer to 'shape-indexed features'
    for(int i = 0;i < candidate_pixel_num;i++){
        double x = random_generator.uniform(-1.0,1.0);
        double y = random_generator.uniform(-1.0,1.0);
        if(x*x + y*y > 1.0){
            i--;
            continue;
        }
        // find nearest landmark index
        double min_dist = 1e10;
        int min_index = 0;
        for(int j = 0;j < mean_shape.rows;j++){
            double temp = pow(mean_shape(j,0)-x,2.0) + pow(mean_shape(j,1)-y,2.0);
            if(temp < min_dist){
                min_dist = temp;
                min_index = j;
            }
        }
        candidate_pixel_locations(i,0) = x - mean_shape(min_index,0);
        candidate_pixel_locations(i,1) = y - mean_shape(min_index,1);
        nearest_landmark_index(i) = min_index;   
    }

    // get densities of candidate pixels for each image
    // for densities: each row is the pixel densities at each candidate pixels for an image 
    // Mat_<double> densities(images.size(), candidate_pixel_num);
    vector<vector<double> > densities;
    densities.resize(candidate_pixel_num);
    for(int i = 0;i < images.size();i++){
        Mat_<double> rotation;
        double scale;
        Mat_<double> temp = ProjectShape(current_shapes[i],bounding_box[i]);
        SimilarityTransform(temp,mean_shape,rotation,scale);
        for(int j = 0;j < candidate_pixel_num;j++){
            double project_x = rotation(0,0) * candidate_pixel_locations(j,0) + rotation(0,1) * candidate_pixel_locations(j,1);
            double project_y = rotation(1,0) * candidate_pixel_locations(j,0) + rotation(1,1) * candidate_pixel_locations(j,1);
            project_x = scale * project_x * bounding_box[i].width / 2.0;
            project_y = scale * project_y * bounding_box[i].height / 2.0;
            int index = nearest_landmark_index(j);
            int real_x = project_x + current_shapes[i](index,0);
            int real_y = project_y + current_shapes[i](index,1); 
            real_x = std::max(0.0,std::min((double)real_x,images[i].cols-1.0));
            real_y = std::max(0.0,std::min((double)real_y,images[i].rows-1.0));
            densities[j].push_back((int)images[i](real_y,real_x));
        }
    }
        
    // calculate the covariance between densities at each candidate pixels 
    Mat_<double> covariance(candidate_pixel_num,candidate_pixel_num);
    Mat_<double> mean;
    for(int i = 0;i < candidate_pixel_num;i++){
        for(int j = i;j< candidate_pixel_num;j++){
            double correlation_result = calculate_covariance(densities[i],densities[j]);
            covariance(i,j) = correlation_result;
            covariance(j,i) = correlation_result;
        }
    } 


    // train ferns
    vector<Mat_<double> > prediction;
    prediction.resize(regression_targets.size());
    for(int i = 0;i < regression_targets.size();i++){
        prediction[i] = Mat::zeros(mean_shape.rows,2,CV_64FC1); 
    } 
    ferns_.resize(second_level_num);
    clock_t t = clock();
    for(int i = 0;i < second_level_num;i++){
        vector<Mat_<double> > temp = ferns_[i].Train(densities,covariance,candidate_pixel_locations,nearest_landmark_index,regression_targets,fern_pixel_num);     
        // update regression targets
        for(int j = 0;j < temp.size();j++){
            prediction[j] = prediction[j] + temp[j];
            regression_targets[j] = regression_targets[j] - temp[j];
        }  
        if((i+1) % 50 == 0){
            cout<<"Fern cascades: "<< curr_level_num << " out of "<< first_level_num<<"; "; 
            cout<<"Ferns: "<<i+1<<" out of "<<second_level_num<<endl;
            double remaining_level_num= (first_level_num - curr_level_num) * 500 + second_level_num - i; 
            double time_remaining = 0.02 * double(clock() - t)  / CLOCKS_PER_SEC * remaining_level_num;
            cout<<"Expected remaining time: "
                << (int)time_remaining / 60<<"min "<<(int)time_remaining % 60 <<"s"<<endl; 
            t = clock();
        }
    }
    
    for(int i = 0;i < prediction.size();i++){
        Mat_<double> rotation;
        double scale;
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_box[i]),mean_shape,rotation,scale);
        transpose(rotation,rotation);
        prediction[i] = scale * prediction[i] * rotation; 
    } 
    return prediction;    
}

void FernCascade::Read(ifstream& fin){
    fin>>second_level_num_; 
    ferns_.resize(second_level_num_);
    for(int i = 0;i < second_level_num_;i++){
        ferns_[i].Read(fin);
    }
}

void FernCascade::Write(ofstream& fout){
    fout<<second_level_num_<<endl;
    for(int i = 0;i < second_level_num_;i++){
        ferns_[i].Write(fout);
    }   
}


Mat_<double> FernCascade::Predict(const Mat_<uchar>& image, 
                          const BoundingBox& bounding_box, 
                          const Mat_<double>& mean_shape,
                          const Mat_<double>& shape){   
    Mat_<double> result = Mat::zeros(shape.rows,2,CV_64FC1);
    Mat_<double> rotation;
    double scale;
    SimilarityTransform(ProjectShape(shape,bounding_box),mean_shape,rotation,scale);
    for(int i = 0;i < second_level_num_;i++){
        result = result + ferns_[i].Predict(image,shape,rotation,bounding_box,scale); 
    }
    SimilarityTransform(ProjectShape(shape,bounding_box),mean_shape,rotation,scale);
    transpose(rotation,rotation);
    result = scale * result * rotation; 
    
    return result; 
}

Mat_<double> GetMeanShape(const vector<Mat_<double> >& shapes,
                          const vector<BoundingBox>& bounding_box){
    Mat_<double> result = Mat::zeros(shapes[0].rows,2,CV_64FC1);
    for(int i = 0;i < shapes.size();i++){
        result = result + ProjectShape(shapes[i],bounding_box[i]);
    }
    result = 1.0 / shapes.size() * result;

    return result;
}

Mat_<double> ProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0)-bounding_box.centroid_x) / (bounding_box.width / 2.0);
        temp(j,1) = (shape(j,1)-bounding_box.centroid_y) / (bounding_box.height / 2.0);  
    } 
    return temp;  
}

Mat_<double> ReProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        temp(j,1) = (shape(j,1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    } 
    return temp; 
}


void SimilarityTransform(const Mat_<double>& shape1, const Mat_<double>& shape2, 
                         Mat_<double>& rotation,double& scale){
    rotation = Mat::zeros(2,2,CV_64FC1);
    scale = 0;
    
    // center the data
    double center_x_1 = 0;
    double center_y_1 = 0;
    double center_x_2 = 0;
    double center_y_2 = 0;
    for(int i = 0;i < shape1.rows;i++){
        center_x_1 += shape1(i,0);
        center_y_1 += shape1(i,1);
        center_x_2 += shape2(i,0);
        center_y_2 += shape2(i,1); 
    }
    center_x_1 /= shape1.rows;
    center_y_1 /= shape1.rows;
    center_x_2 /= shape2.rows;
    center_y_2 /= shape2.rows;
    
    Mat_<double> temp1 = shape1.clone();
    Mat_<double> temp2 = shape2.clone();
    for(int i = 0;i < shape1.rows;i++){
        temp1(i,0) -= center_x_1;
        temp1(i,1) -= center_y_1;
        temp2(i,0) -= center_x_2;
        temp2(i,1) -= center_y_2;
    }

     
    Mat_<double> covariance1, covariance2;
    Mat_<double> mean1,mean2;
    // calculate covariance matrix
    calcCovarMatrix(temp1,covariance1,mean1,CV_COVAR_COLS);
    calcCovarMatrix(temp2,covariance2,mean2,CV_COVAR_COLS);

    double s1 = sqrt(norm(covariance1));
    double s2 = sqrt(norm(covariance2));
    scale = s1 / s2; 
    temp1 = 1.0 / s1 * temp1;
    temp2 = 1.0 / s2 * temp2;

    double num = 0;
    double den = 0;
    for(int i = 0;i < shape1.rows;i++){
        num = num + temp1(i,1) * temp2(i,0) - temp1(i,0) * temp2(i,1);
        den = den + temp1(i,0) * temp2(i,0) + temp1(i,1) * temp2(i,1);      
    }
    
    double norm = sqrt(num*num + den*den);    
    double sin_theta = num / norm;
    double cos_theta = den / norm;
    rotation(0,0) = cos_theta;
    rotation(0,1) = -sin_theta;
    rotation(1,0) = sin_theta;
    rotation(1,1) = cos_theta;
}

double calculate_covariance(const vector<double>& v_1, 
                            const vector<double>& v_2){
    Mat_<double> v1(v_1);
    Mat_<double> v2(v_2);
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v2)[0];
    v1 = v1 - mean_1;
    v2 = v2 - mean_2;
    return mean(v1.mul(v2))[0]; 
}

@implementation ShapeRegressorWrapper

- (id)initWithModelFromPath:(NSString *)modelPath
{
    self = [super init];

    _cppRegressor = new ShapeRegressor();
    _cppRegressor->Load(std::string([modelPath UTF8String]));

    return self;
}

- (NSArray *)predictFrame:(Frame *)frame withFaceRect:(NSRect)rect
{
    int initial_number = 20;

    Mat frameImage(frame.cvSourceImage, false);
    BoundingBox bb;
    bb.start_x = rect.origin.x;
    bb.start_y = rect.origin.y;
    bb.width = rect.size.width;
    bb.height = rect.size.height;
    bb.centroid_x = bb.start_x + bb.width/2.0;
    bb.centroid_y = bb.start_y + bb.height/2.0;

    Mat_<double> current_shape = _cppRegressor->Predict(frameImage, bb, initial_number);

    int nLandmarks = 29;
    NSMutableArray *landmarksArray
        = [[NSMutableArray alloc] initWithCapacity:nLandmarks];
    for (int i = 0; i < nLandmarks; i++) {
        NSPoint landmarkPoint = {
            current_shape(i,0), current_shape(i,2)
        };
        [landmarksArray addObject:([NSValue valueWithPoint:landmarkPoint])];
    }

    return landmarksArray;
}


- (void)dealloc
{
    delete _cppRegressor;
    [super dealloc];
}

@end
// #endif