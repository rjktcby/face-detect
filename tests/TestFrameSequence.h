#ifndef __FACE_DETECT_TEST_FRAME_SEQUENCE_H__
#define __FACE_DETECT_TEST_FRAME_SEQUENCE_H__

#import <Foundation/Foundation.h>
#import <OpenCVContext.h>
#import <Capture.h>

// TestFrameSequence - class for testing sequence of frames from files

@interface TestFrameSequence : NSObject {
    OpenCVContext *_cvContext;
    Capture *_capture;
    int _currentFrame;
    NSArray *_framePaths;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withArguments:(NSArray *)args;
- (Frame *)nextFrame;
- (bool)finished;

@end

#endif // __FACE_DETECT_TEST_FRAME_SEQUENCE_H__
