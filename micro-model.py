TF_LITE_MICRO_TESTS_BEGIN
//set up andd load the model

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
    tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}

//memory in the chip allocaction
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
