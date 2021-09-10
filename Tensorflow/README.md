# General Information

Tensorflow is a library for Machine Learning applications

# Build for arm

Information: This build is based on the cross-compiler toolchain setup from [gsdr-system](https://gitlab.dlr.de/adsb-receiver/gsdr-system). The skript to build the cross-compiler toolchain is located in Skripte/Cross-Compiler.

Download tensorflow via:

```
$git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

## Lite (C/C++)

1. Go into the make file folder:

```
$cd tensorflow_src/tensorflow/lite/tools/make
```

2. Download additional dependencies:

```
$./download_dependencies.sh
```

3. Modify the build file (last line):

```
$cp build_rpi_lib.sh build_arm_lib.sh
$nano build_arm_lib.sh
(-) make -j ${NO_JOB} TARGET=rpi -C "${TENSORFLOW_DIR}" -f tensorflow/lite/tools/make/Makefile $@
(+) make CC=arm-linux-gnueabihf-gcc CCX=arm-linux-gnueabihf-g++ -j 2 TARGET=arm -C "${TENSORFLOW_DIR}" -f tensorflow/lite/tools/make/Makefile $@
```

4. Modify the makefile:

```
$nano Makefile
$LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz \
-ldl \
(+) -latomic \
(+) -lrt \

$CC_PREFIX := arm-linux-gnueabihf-
```

5. Modify the util.cpp:

```
$nano downloads/flatbuffers/src/util.cpp
(-) #include <limits.h>
(+) #include <linux/limits.h>
```

6. Run the build file:

```
$./build_arm_lib.sh
```
