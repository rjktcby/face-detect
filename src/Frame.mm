#import "Frame.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>
#import <opencv/ml.h>

@interface Frame()
+ (IplImage *)loadImageFromFile:(NSString *)imagePath;
+ (IplImage *)makeGrayImageFrom:(IplImage *)cvSourceImage;
@end

@implementation Frame

@synthesize cvSourceImage = cvSourceImage_;
@synthesize cvGrayImage = cvGrayImage_;

+ (IplImage *)makeGrayImageFrom:(IplImage *)cvSourceImage
{
    NSAssert(cvSourceImage, @"Attempt to make gray image from nil");

    IplImage *cvGrayImage = cvCreateImage(cvGetSize(cvSourceImage), IPL_DEPTH_8U, 1);
    cvCvtColor (cvSourceImage, cvGrayImage, CV_BGR2GRAY);
    cvEqualizeHist (cvGrayImage, cvGrayImage);

    return cvGrayImage;
}

+ (IplImage *)loadImageFromFile:(NSString *)imagePath
{
    IplImage *cvImage = cvLoadImage([imagePath cString]);
    NSAssert(cvImage, @"Unable to open image path");

    return cvImage;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withMemory:(const char *)memory withSize:(size_t)size;
{
    self = [super init];

    cvContext_ = cvContext;
    //TODO

    return self;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withPath:(NSString *)imagePath
{
    self = [super init];

    cvContext_ = cvContext;
    cvSourceImage_ = [Frame loadImageFromFile:imagePath];
    cvGrayImage_ = [Frame makeGrayImageFrom:cvSourceImage_];

    return self;
}

- (void)dealloc
{
    cvReleaseImage(&cvGrayImage_);
    cvReleaseImage(&cvSourceImage_);

    [super dealloc];
}
@end
