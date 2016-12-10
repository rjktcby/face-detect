CC        := clang++-3.5
LD        := gcc

SRC_DIR   := src src/tests
BUILD_DIR := build build/tests

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.mm))
OBJ       := $(patsubst src/%.mm,build/%.o,$(SRC))
INCLUDES  := $(addprefix -I,$(SRC_DIR))

CFLAGS    := -ObjC++ --std=c++11 `gnustep-config --objc-flags` -fobjc-runtime=gnustep-1.7 -I/usr/lib/gcc/x86_64-linux-gnu/4.9/include -fdiagnostics-color=always
LDFLAGS   := -L/usr/local/lib -lgnustep-base -lobjc -ldlib -llapack -lm -lstdc++ `pkg-config opencv --libs` -fdiagnostics-color=always

vpath %.mm $(SRC_DIR)

define make-goal
$1/%.o: %.mm
	$(CC) $(CFLAGS) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs build/face-detect

build/face-detect: $(OBJ)
	$(LD) $(LDFLAGS) $^ -o $@


checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
