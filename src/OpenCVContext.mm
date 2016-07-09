#import "OpenCVContext.h"

#import <opencv/cv.h>

@interface OpenCVContext()

- (CvHaarClassifierCascade *)getHaarCascadeWithName:(NSString *)haarCascadeName;

@end

@implementation OpenCVContext

- (id)initWithOpenCVPath:(NSString *)openCVPath
{
    self = [super init];

    openCVPath_ = openCVPath;
    haarCascadesPath_ = [NSString stringWithFormat:@"%@/haarcascades", openCVPath_];
    haarCascades_ = [[NSMutableDictionary alloc] init];

    cvStorage_ = cvCreateMemStorage(0);
    cvClearMemStorage(cvStorage_);

    return self;
}

- (void)dealloc
{
    [haarCascades_ release];
    cvReleaseMemStorage(&cvStorage_);
    [super dealloc];
}

- (CvHaarClassifierCascade *)getHaarCascadeWithName:(NSString *)haarCascadeName
{
    if (![haarCascades_ objectForKey:haarCascadeName]) {

        NSString *haarCascadePath = [NSString stringWithFormat:@"%@/%@", haarCascadesPath_, haarCascadeName];
        NSLog(@"Loading haar cascade from %@", haarCascadePath);

        CvHaarClassifierCascade *cvHaarCascade
            = (CvHaarClassifierCascade *) cvLoad([haarCascadePath cString], 0, 0, 0);
        NSAssert(cvHaarCascade, ([NSString stringWithFormat:@"Can't load haar cascade, path=%@", haarCascadePath]));

        [haarCascades_ setValue:[NSValue valueWithPointer:cvHaarCascade] forKey:haarCascadeName];
    }

    return (CvHaarClassifierCascade *)[[haarCascades_ valueForKey:haarCascadeName] pointerValue];
}

- (NSArray *)detectObjectsInGrayImage:(IplImage *)cvGrayImage withCascadeName:(NSString *)haarCascadeName
{
    NSAssert(cvGrayImage, @"Can't detect objects on image=nil");

    CvHaarClassifierCascade *cvHaarCascade = [self getHaarCascadeWithName:haarCascadeName];
    NSAssert(cvHaarCascade, ([NSString stringWithFormat:@"Haar cascade is nil, name=%@", haarCascadeName]));

    //TODO get rid of magic numbers
    CvSeq *cvObjects= cvHaarDetectObjects(cvGrayImage, cvHaarCascade,
        cvStorage_, 1.11, 4, 0, cvSize(40, 40));

    int nObjects = cvObjects ? cvObjects->total : 0;
    if (nObjects == 0) {
        return nil;
    }

    NSMutableArray *rectsArray = [[NSMutableArray alloc] initWithCapacity:nObjects];
    for (unsigned i = 0; i < nObjects; i++) {
        CvRect *r = (CvRect *) cvGetSeqElem (cvObjects, i);
        NSRect objectRect = {
            { r->x, r->y },             // point
            { r->width, r->height }     // size
        };
        [rectsArray addObject:([NSValue valueWithRect:objectRect])];
    }
    return rectsArray;
}

@end