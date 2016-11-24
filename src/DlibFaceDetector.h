#ifndef __FACE_DETECT_DLIB_FACE_DETECTOR_H__
#define __FACE_DETECT_DLIB_FACE_DETECTOR_H__

#import <Foundation/Foundation.h>

#import <OpenCVContext.h>
#import <Frame.h>

// DlibFaceDetector - face detector based on dlib's frontal_face_detector

@interface DlibFaceDetector : NSObject

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext;
- (NSArray *)detectFacesInFrame:(Frame *)frame;
@end

#endif // __FACE_DETECT_DLIB_FACE_DETECTOR_H__