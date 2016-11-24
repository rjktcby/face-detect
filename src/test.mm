#import <Foundation/Foundation.h>

@interface Test : NSObject
- (id)init;
@end

@implementation Test
{
    void *_ivar;
}

- (id)init
{
    self = [super init];

    _ivar = new int();

    NSString *str = [NSString stringWithUTF8String:"Hello"];
    int len = [str length];
    NSLog(@"%@", str);

    return self;
}

- (void)dealloc
{
    delete (int *)_ivar;

    [super dealloc];
}

@end

int main()
{
    NSAutoreleasePool *arPool = [[NSAutoreleasePool alloc] init];
    Test *test = [[Test new] autorelease];

    [arPool drain];
}