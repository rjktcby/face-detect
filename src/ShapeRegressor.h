#ifndef __FACE_DETECT_SHAPE_REGRESSOR_H__
#define __FACE_DETECT_SHAPE_REGRESSOR_H__

#import <Foundation/Foundation.h>

class ShapeRegressor;

@interface ShapeRegressorWrapper : NSObject
{
    ShapeRegressor *_cppRegressor;
}

- (id)initWithModelFromPath:(NSString *)modelPath;

@end

#endif // __FACE_DETECT_SHAPE_REGRESSOR_H__
