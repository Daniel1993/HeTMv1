#ifndef INPUT_HANDLER_H_GUARD
#define INPUT_HANDLER_H_GUARD

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

  void input_parse(int argc, char **argv);
  long input_getLong(const char *arg);
  double input_getDouble(const char *arg);
  size_t input_getString(const char *arg, char *out);
  int input_exists(const char *arg);

  // loops all inputs, return !=0 to stop
  int input_foreach(int(*fn)(const char*));

#ifdef __cplusplus
}
#endif

#endif /* INPUT_HANDLER_H_GUARD */
