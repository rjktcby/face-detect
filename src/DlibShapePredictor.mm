#import "DlibShapePredictor.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>
#import <opencv/ml.h>

#import <dlib/threads/thread_pool_extension.h>
// #import <dlib/image_processing/shape_predictor.h>
#import <dlib/opencv/cv_image.h>

@implementation DlibShapePredictor
{
@private
    OpenCVContext *_cvContext;
    // dlib::shape_predictor _dlibShapePredictor;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withModelPath:(NSString *)modelPath
{
    self = [super init];

    _cvContext = cvContext;

    try {
        // dlib::deserialize([modelPath UTF8String]) >> _dlibShapePredictor;
    } catch (dlib::serialization_error& e) {
        NSLog(@"Unable to deserialize model from path=%@", modelPath);
    }

    return self;
}

- (void)dealloc
{
    [super dealloc];
}

#if 0
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
#endif

- (void)predictShapeForFrame:(Frame *)frame forFaceRect:(NSRect)faceRect
{
    cv::Mat frameImage(frame.cvSourceImage, false);
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frameImage);

    dlib::rectangle dlibRect(
        faceRect.origin.x,
        faceRect.origin.y,
        faceRect.origin.x + faceRect.size.width,
        faceRect.origin.y + faceRect.size.height
    );

    // dlib::full_object_detection dlibShape = _dlibShapePredictor(dlibImg, dlibRect);
}

@end
