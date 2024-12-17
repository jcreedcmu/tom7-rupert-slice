
#include "ansi-image.h"

#include <memory>
#include <cstdio>

#include "ansi.h"
#include "base/logging.h"
#include "image.h"

static void Favicon() {
  std::unique_ptr<ImageRGBA> favicon(ImageRGBA::Load("favicon.png"));

  // The characters have a black outline.
  printf("%s", ANSIImage::HalfChar(*favicon, 0xAAAAAAFF).c_str());



  printf("^ This test requires visual inspection.\n");
}


int main(int argc, char **argv) {
  ANSI::Init();

  Favicon();


  printf("OK\n");
  return 0;
}
