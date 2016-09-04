#import <OpenCVContext.h>
#import <Frame.h>

#import <opencv/cv.h>
#import <dlib/image_processing/frontal_face_detector.h>
#import <dlib/gui_widgets.h>
#import <dlib/opencv/cv_image.h>

@interface OpenCVContext()

- (CvHaarClassifierCascade *)getHaarCascadeWithName:(NSString *)haarCascadeName;
- (NSArray *)detectObjectsInGrayImage:(IplImage *)cvGrayImage withCascadeName:(NSString *)cascadeName;

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
            { double(r->x), double(r->y) },             // point
            { double(r->width), double(r->height) }     // size
        };
        [rectsArray addObject:([NSValue valueWithRect:objectRect])];
    }
    return rectsArray;
}

- (NSArray *)detectFacesWithOpenCVInFrame:(Frame *)frame
{
    return [self detectObjectsInGrayImage:frame.cvGrayImage
                          withCascadeName:@"haarcascade_frontalface_default.xml"];
}

- (NSArray *)detectFacesWithDLibInFrame:(Frame *)frame
{
    cv::Mat frameImage(frame.cvGrayImage, false);

    dlib::frontal_face_detector dlibDetector = dlib::get_frontal_face_detector();
    std::vector<dlib::rectangle> faceRects = dlibDetector(dlib::cv_image<uchar>(frameImage));

    int nRects = faceRects.size();
    NSMutableArray *rectsArray = [[NSMutableArray alloc] initWithCapacity:nRects];
    for (unsigned i = 0; i < nRects; i++) {
        dlib::rectangle &rect = faceRects[i];
        NSRect objectRect = {
            { double(rect.left()), double(rect.top()) },        // point
            { double(rect.width()), double(rect.height()) }     // size
        };
        [rectsArray addObject:([NSValue valueWithRect:objectRect])];
    }
    return rectsArray;
}

@end