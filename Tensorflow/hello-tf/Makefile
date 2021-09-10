# Define target
TARGET_EXEC := payload

# Define (cross-)compiler and flags
CC = arm-linux-gnueabihf-gcc
#CC = gcc
CFLAGS = -O3 -mfpu=neon -mcpu=cortex-a9 -mfloat-abi=hard

CXX = arm-linux-gnueabihf-g++
#CXX = g++
CXXFLAGS = ${CFLAGS} --std=c++11

# Define external libraries
LIBFLAGS := -L./
LDFLAGS := -lstdc++ \
-lpthread \
-lm \
-ldl \
-lz \
-latomic \
-ltensorflow-lite

# Setup build environment
BUILD_DIR := ./build
SRC_DIRS := ./src

# Find all the C and C++ we want to compile
SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c)

# String subsitution for every C/C++ file
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution
DEPS := $(OBJS:.o=.d)

# Pass header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d) \
../tensorflow \
../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \

#prefix to INC_DIRS
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# -MMD and -MP flags for Makefiles
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# Final build step
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBFLAGS) $(LDFLAGS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
