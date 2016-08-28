#import "Capture.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>

// @interface Capture()
// @end

@implementation Capture

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext;
{
    self = [super init];

    cvContext_ = cvContext;
    cvCapture_ = cvCreateCameraCapture(0);

    if (!cvCapture_) {
        NSLog(@"Could not create camera capture");
        return nil;
    } else {
        NSLog(@"Created camera capture");
    }

    return self;
}

- (void)dealloc
{
    cvReleaseCapture(&cvCapture_);

    [super dealloc];
}

- (Frame *)grabFrame
{
    NSAssert(cvCapture_, @"cvCapture not initialized");

    Frame *frame = [[Frame alloc] initWithOpenCVContext:cvContext_
                                    withImage:cvQueryFrame(cvCapture_)];
    return frame;
}

@end
