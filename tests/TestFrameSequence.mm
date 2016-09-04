#import "TestFrameSequence.h"

#import <opencv/cv.h>
#import <opencv/highgui.h>

using namespace cv;

@implementation TestFrameSequence

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withArguments:(NSArray *)args;
{
    self = [super init];

    NSAssert([args count] != 0, @"Need at least one frame path in arguments");

    _cvContext = cvContext;
    _framePaths = args;
    _currentFrame = 0;

    return self;
}

- (void)dealloc
{
    [super dealloc];
}

- (Frame *)nextFrame
{
    NSString *framePath = [_framePaths objectAtIndex:_currentFrame];
    NSLog(@"Loading frame from path=%@", framePath);

    Frame *frame = [[Frame alloc] initWithOpenCVContext:_cvContext withPath:framePath];

    _currentFrame++;
    if (_currentFrame > [_framePaths count]) {
        _currentFrame = 0;
    }

    return frame;
}

- (bool)finished
{
    return NO;
}

@end
