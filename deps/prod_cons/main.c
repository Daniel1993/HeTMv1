#include "prod-cons.h"

#include <stdlib.h>
#include <stdio.h>

static void callback(void *arg) {
  printf("Hello from other thread! (%p)\n", arg);
}

int main () {

  prod_cons_start_thread();

  // sends three requests, note the requests are serialized
  for (long i = 0; i < 0x200; ++i) {
    prod_cons_async_request((prod_cons_async_req_s){
      .args = (void*)i,
      .fn = callback
    });
  }

  prod_cons_join_thread();

  return EXIT_SUCCESS;
}
