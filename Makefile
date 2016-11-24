CC        := clang++-3.9
LD        := clang++-3.9

SRC_DIR   := src src/tests
BUILD_DIR := build

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.mm))
OBJ       := $(patsubst src/%.mm,build/%.o,$(SRC))
INCLUDES  := $(addprefix -I,$(SRC_DIR))

CFLAGS    := -ObjC++ --std=c++11 `gnustep-config --objc-flags` -fobjc-runtime=gnustep-1.7
LDFLAGS   := -lgnustep-base -lobjc -ldlib -llapack `pkg-config opencv --libs` `gnustep-config --objc-libs`

vpath %.mm $(SRC_DIR)

define make-goal
$1/%.o: %.mm
	$(CC) $(CFLAGS) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs build/face-detect

build/face-detect: $(OBJ)
	$(LD) $(CFLAGS) $(LDFLAGS) $^ -o $@


checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
