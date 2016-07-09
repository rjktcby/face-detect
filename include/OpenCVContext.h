#ifndef __FACE_DETECT_OPENCV_CONTEXT_H__
#define __FACE_DETECT_OPENCV_CONTEXT_H__

#import <Foundation/Foundation.h>
#import <OpenCVForwardDeclarations.h> // IplImage

// OpenCVContext - class for common opencv stuff (TBD)

@interface OpenCVContext : NSObject {
    NSString *openCVPath_, *haarCascadesPath_;
    CvMemStorage *cvStorage_;
    NSMutableDictionary *haarCascades_;
}

-(id)initWithOpenCVPath:(NSString *)openCVPath;

- (NSArray *)detectObjectsInGrayImage:(IplImage *)cvGrayImage withCascadeName:(NSString *)cascadeName;

@end

#endif // __FACE_DETECT_OPENCV_CONTEXT_H__