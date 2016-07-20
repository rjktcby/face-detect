#ifndef __FACE_DETECT_FRAME_H__
#define __FACE_DETECT_FRAME_H__

#import <Foundation/Foundation.h>

#import <OpenCVForwardDeclarations.h> // IplImage
#import <OpenCVContext.h>

// Frame - class incapsulating frame image and metainfo (detected faces, etc)

@interface Frame : NSObject {
    OpenCVContext *cvContext_;
    IplImage *cvSourceImage_, *cvGrayImage_;
}

@property(nonatomic) IplImage *cvSourceImage,
                              *cvGrayImage;

- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withMemory:(const char *)memory withSize:(size_t)size;
- (id)initWithOpenCVContext:(OpenCVContext *)cvContext withPath:(NSString *)filename;

@end

#endif // __FACE_DETECT_FRAME_H__
