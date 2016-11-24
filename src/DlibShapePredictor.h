#ifndef __FACE_DETECT_DLIB_SHAPE_PREDICTOR_H__
#define __FACE_DETECT_DLIB_SHAPE_PREDICTOR_H__

#import <Foundation/Foundation.h>

#import <OpenCVContext.h>
#import <Frame.h>

// DlibShapePredictor - shape predictor based on dlib's shape_predictor

@interface DlibShapePredictor : NSObject
- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withModelPath:(NSString *)madelPath;
- (void)predictShapeForFrame:(Frame *)frame forFaceRect:(NSRect)faceRect;


// - (NSArray *)detectFacesInFrame:(Frame *)frame;

@end

#endif // __FACE_DETECT_DLIB_SHAPE_PREDICTOR_H__