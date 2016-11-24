#import "DlibFaceDetector.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>
#import <opencv/ml.h>

#import <dlib/image_processing/frontal_face_detector.h>
#import <dlib/opencv/cv_image.h>

@implementation DlibFaceDetector
{
@private
    OpenCVContext *_cvContext;
    dlib::frontal_face_detector _dlibDetector;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext
{
    self = [super init];

    _cvContext = cvContext;
    _dlibDetector = dlib::frontal_face_detector();

    return self;
}

- (void)dealloc
{
    [super dealloc];
}

- (NSArray *)detectFacesInFrame:(Frame *)frame
{
    cv::Mat frameImage(frame.cvGrayImage, false);

    std::vector<dlib::rectangle> faceRects = _dlibDetector(dlib::cv_image<uchar>(frameImage));

    int nRects = faceRects.size();
    NSMutableArray *rectsArray = [[NSMutableArray alloc] initWithCapacity:nRects];
    for (unsigned i = 0; i < nRects; i++) {
        dlib::rectangle &rect = faceRects[i];
        NSRect objectRect = {
            { double(rect.left()), double(rect.top()) },        // point
            { double(rect.width()), double(rect.height()) }     // size
        };
        [rectsArray addObject:([NSValue valueWithRect:objectRect])];
    }
    return rectsArray;
}

@end
