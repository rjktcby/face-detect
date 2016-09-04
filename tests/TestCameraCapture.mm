#import "TestCameraCapture.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>

using namespace cv;

@implementation TestCameraCapture

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withArguments:(NSArray *)args;
{
    self = [super init];

    _cvContext = cvContext;
    _capture = [[Capture alloc] initWithOpenCVContext:cvContext];
    NSAssert(_capture != nil, @"Failed to create camera capture");

    return self;
}

- (void)dealloc
{
    [super dealloc];
}

- (Frame *)nextFrame
{
    return [_capture grabFrame];
}

- (bool)finished
{
    return NO;
}

@end
