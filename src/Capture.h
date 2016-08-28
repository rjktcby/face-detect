#ifndef __FACE_DETECT_CAPTURE_H__
#define __FACE_DETECT_CAPTURE_H__

#import <Foundation/Foundation.h>

#import <OpenCVForwardDeclarations.h> // CvCapture
#import <OpenCVContext.h>
#import <Frame.h>

// Capture - class incapsulating frame capture from camera

@interface Capture : NSObject {
    OpenCVContext *cvContext_;
    CvCapture *cvCapture_;
}

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext;
- (Frame *)grabFrame;
@end

#endif // __FACE_DETECT_CAPTURE_H__
