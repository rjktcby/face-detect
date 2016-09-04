#import <TestCameraCapture.h>
#import <TestFrameSequence.h>
#import <ShapeRegressor.h>

#import <opencv/cv.h>
#import <opencv/highgui.h>

using namespace cv;

int main (int argc, char **argv)
{
    if (argc < 3) {
        NSLog(@"Usage: ./%s <path to shape regression model> <test name> <args>", argv[0]);
        return 0;
    }

    NSAutoreleasePool *arPool = [[NSAutoreleasePool alloc] init];
    OpenCVContext *cvContext = [[OpenCVContext alloc] initWithOpenCVPath:@"/usr/share/opencv"];

    NSString *testName = [NSString stringWithUTF8String:argv[2]];

    NSMutableArray *testArgs = [NSMutableArray arrayWithCapacity:(argc-3)];
    for (int i = 0; i < argc-3; i++) {
        [testArgs addObject:[NSString stringWithUTF8String:argv[i+3]]];
    }

    NSLog(@"Running test=%@ args=%@", testName, testArgs);

    cvNamedWindow ([testName UTF8String], CV_WINDOW_AUTOSIZE);

    static NSDictionary *testClasses = @{
        @"TestCameraCapture": [TestCameraCapture class],
        @"TestFrameSequence": [TestFrameSequence class]};

    Class *testClass = (Class *)[testClasses objectForKey:testName];
    if (!testClass) {
        NSLog(@"invalid test=%@", testName);
        return 0;
    }

    id test = [[testClass alloc] initWithOpenCVContext:cvContext withArguments:testArgs];

    ShapeRegressorWrapper *regressor
        = [[ShapeRegressorWrapper alloc] initWithModelFromPath:[NSString stringWithUTF8String:argv[1]]];

    int frame_num = 0;
    while (![test finished]) {
        NSLog(@"Grabbing frame #%d", frame_num);
        Frame *frame = [test nextFrame];

        NSArray *frameFaces = [cvContext detectObjectsInGrayImage:frame.cvGrayImage
                                         withCascadeName:@"haarcascade_frontalface_default.xml"];
        NSLog(@"Found %d frontal faces", [frameFaces count]);

        Mat frameImage(frame.cvSourceImage, false);
        for(int i = 0; i < [frameFaces count]; i++)
        {
            NSRect faceRect = [[frameFaces objectAtIndex:i] rectValue];

            NSArray *landmarks = [regressor predictFrame:frame withFaceRect:faceRect];

            line(frameImage,Point2d(faceRect.origin.x, faceRect.origin.y),
                Point2d(faceRect.origin.x+faceRect.size.width, faceRect.origin.y),Scalar(255,0,0));
            line(frameImage,Point2d(faceRect.origin.x+faceRect.size.width, faceRect.origin.y),
                Point2d(faceRect.origin.x+faceRect.size.width, faceRect.origin.y+faceRect.size.height),Scalar(255,0,0));
            line(frameImage,Point2d(faceRect.origin.x+faceRect.size.width, faceRect.origin.y+faceRect.size.height),
                Point2d(faceRect.origin.x, faceRect.origin.y+faceRect.size.height),Scalar(255,0,0));
            line(frameImage,Point2d(faceRect.origin.x, faceRect.origin.y+faceRect.size.height),
                Point2d(faceRect.origin.x, faceRect.origin.y),Scalar(255,0,0));

            for (int l = 0; l < [landmarks count]; l++) {
                NSPoint landmarkPoint = [[landmarks objectAtIndex:l] pointValue];
                circle(frameImage, Point2d(landmarkPoint.x, landmarkPoint.y), 3,
                       Scalar(255,0,0), -1, 8, 0);
            }
        }

        cvShowImage([testName UTF8String], frame.cvSourceImage);

        int c = cvWaitKey (2);
        if (c == '\x1b') {
            break;
        }
        frame_num++;
    }

    [arPool drain];

    cvDestroyWindow ("Capture");

    return 1;
}