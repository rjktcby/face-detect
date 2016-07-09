#import "Frame.h"
#import "OpenCVContext.h"

int main (int argc, char **argv)
{
    if (argc <= 1) {
        NSLog(@"usage: %s <path to frame 1> .. <path to frame N>");
        return 0;
    }

    NSAutoreleasePool *arPool = [[NSAutoreleasePool alloc] init];
    OpenCVContext *cvContext = [[OpenCVContext alloc] initWithOpenCVPath:@"/usr/share/opencv"];

    for (int i = 1; i < argc; i++) {
        NSString *framePath = [NSString stringWithUTF8String:argv[i]];

        NSLog(@"Loading frame path=%@", framePath);
        Frame *frame = [[Frame alloc] initWithOpenCVContext:cvContext withPath:framePath];

        NSArray *frameFaces = [cvContext detectObjectsInGrayImage:frame.cvGrayImage
                                         withCascadeName:@"haarcascade_frontalface_default.xml"];
        NSLog(@"Found %d frontal faces", [frameFaces count]);
    }

    [arPool drain];
    return 0;
}