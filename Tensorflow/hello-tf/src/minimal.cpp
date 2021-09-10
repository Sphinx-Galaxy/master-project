/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ctime>
#include <cstdio>
#include <fstream>
#include <cmath>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

size_t load_from_file(const char * filename, float buffer[]);
void infere_model(std::unique_ptr<tflite::Interpreter> &interpreter, float buffer[], size_t length);
int32_t char_to_int(char buffer[]);
float char_to_float(char buffer[]);
bool check_result(std::unique_ptr<tflite::Interpreter> &interpreter, float boundary);

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <data>\n");
    return 1;
  }
  const char* model_filename = argv[1];
  const char* data_filename = argv[2];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

//	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	clock_t t = clock();
	float buffer[5256*100];
	size_t length = load_from_file(data_filename, buffer);
	infere_model(interpreter, buffer, 5256*100);
//	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	t = clock() - t;

//	printf("Time difference: %llu ns\n", 
//		std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	printf("This operation took: %d clicks\n", t);

  return 0;
}

size_t load_from_file(const char* filename, float buffer[]) {
	std::ifstream fileBuffer(filename, std::ios::in|std::ios::binary);

	if(!fileBuffer.is_open())
		return 0;

    	fileBuffer.seekg(0, std::ios::beg);

	//Read the header
	char header[4];
	fileBuffer.read(header, 4);
	int32_t magic_number = char_to_int(header);
	fileBuffer.read(header, 4);
	int32_t items = char_to_int(header);
	fileBuffer.read(header, 4);
	int32_t rows =  char_to_int(header);
	fileBuffer.read(header, 4);
	int32_t columns =  char_to_int(header);

	printf("Reading file: %s\nMagic Number: %d\nItems: %d\nRows: %d\nColumns: %d\n", filename, magic_number, items, rows, columns);

	//Read the content
	char value[2];
	for(size_t i = 0; i < 5256*100; ++i) {
		fileBuffer.read(value, 2);		
		buffer[i] = char_to_float(value);
	}

	//Test one orbit
	for(size_t i = 0; i < 100; ++i)
		printf("Buffer[%d]: %f\n", i, buffer[i]);

	size_t length = fileBuffer.tellg();
	fileBuffer.close();
	
	return length;
}

void infere_model(std::unique_ptr<tflite::Interpreter> &interpreter, float buffer[], size_t length) {
	int anomaly_count = 0;
	for(size_t i = 0; i < length; ++i) {
		interpreter->typed_input_tensor<float>(0)[i%100] = buffer[i];

		if(i%100 == 99) {
			interpreter->Invoke();
			//tflite::PrintInterpreterState(interpreter.get());
			if(check_result(interpreter, 2.5))
				//printf("Anomaly %d found!\n", ++anomaly_count);
				anomaly_count++;
		}
	}
}

bool check_result(std::unique_ptr<tflite::Interpreter> &interpreter, float boundary) {
	float result = 0;
	for(int i = 0; i < 100; i++) {
		float out = interpreter->typed_output_tensor<float>(0)[i];
		float in = interpreter->typed_input_tensor<float>(0)[i];
		result += std::abs(std::abs(out) - std::abs(in));
		//printf("In: %f Out: %f Diff: %f Result: %f\n", in, out, std::abs(std::abs(out) - std::abs(in)), result);
	}

	//printf("Check result: %f\n", result);

	return result > boundary;
}

int32_t char_to_int(char buffer[]) {
	int32_t result = 0;
	for(size_t i = 0; i < 4; ++i)
		result += buffer[i] << (8*i);

	return result;
}

float char_to_float(char buffer[]) {
	float result = 0;
	for(size_t i = 0; i < 2; ++i)
		result += buffer[i] << (8*i);

	return result / 32768.0; // 2 ** 15
}
