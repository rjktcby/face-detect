#ifndef __FACE_DETECT_TEST_CAMERA_CAPTURE_H__
#define __FACE_DETECT_TEST_CAMERA_CAPTURE_H__

#import <Foundation/Foundation.h>
#import <OpenCVContext.h>
#import <Capture.h>

// TestCameraCapture - class for testing camera captured frames

@interface TestCameraCapture : NSObject {
    OpenCVContext *_cvContext;
    Capture *_capture;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withArguments:(NSArray *)args;
- (Frame *)nextFrame;
- (bool)finished;

@end

#endif // __FACE_DETECT_TEST_CAMERA_CAPTURE_H__
