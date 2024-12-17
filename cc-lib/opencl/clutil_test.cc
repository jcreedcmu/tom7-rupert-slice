
#include "clutil.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstdio>
#include <bit>
#include <vector>

#include "base/logging.h"
#include "base/stringprintf.h"

static CL *cl = nullptr;

static void TestCL() {
  const int BYTES = 1024;
  std::vector<uint8_t> cpu_buffer(BYTES);
  for (int i = 0; i < BYTES; i++) {
    cpu_buffer[i] = std::rotr<uint8_t>(i * 0x2A, 5) ^ 0x11;
  }

  cl_mem gpu_buffer = CreateUninitializedGPUMemory<uint8_t>(cl->context, BYTES);
  CopyBufferToGPU(cl->queue, cpu_buffer, gpu_buffer);

  const auto &[program, kernel] =
    cl->BuildOneKernel(
        R"(
          typedef uchar uint8_t;
          typedef uint uint32_t;
          typedef ulong uint64_t;
          typedef long int64_t;
          typedef atomic_uint atomic_uint32_t;

          // Writes bitmask to out.
          __kernel void Increment(__global uint8_t *restrict mem) {
            const int idx = get_global_id(0);
            mem[idx]++;
          }
        )",
        "Increment",
        0);

  {
    std::vector<uint8_t> readback1 =
      CopyBufferFromGPU<uint8_t>(cl->queue, gpu_buffer, BYTES);
    CHECK(cpu_buffer == readback1);
  }

  // Run Kernel.

  CHECK_SUCCESS(clSetKernelArg(kernel, 0, sizeof (cl_mem),
                               (void *)&gpu_buffer));

  // Simple 1D Kernel
  size_t global_work_offset[] = { (size_t)0 };
  size_t global_work_size[] = { (size_t)BYTES };

  CHECK_SUCCESS(
      clEnqueueNDRangeKernel(cl->queue, kernel,
                             // 1D
                             1,
                             // It does its own indexing
                             global_work_offset,
                             global_work_size,
                             // No local work
                             nullptr,
                             // No wait list
                             0, nullptr,
                             // no event
                             nullptr));

  clFinish(cl->queue);

  {
    std::vector<uint8_t> readback2 =
      CopyBufferFromGPU<uint8_t>(cl->queue, gpu_buffer, BYTES);
    CHECK(readback2.size() == BYTES);

    for (int i = 0; i < BYTES; i++) {
      CHECK(readback2[i] == cpu_buffer[i] + 1) << i;
    }
  }

  CHECK_SUCCESS(clReleaseKernel(kernel));
  CHECK_SUCCESS(clReleaseProgram(program));

  CHECK_SUCCESS(clReleaseMemObject(gpu_buffer));
  printf("Basic kernel OK.\n");
}

int main(int argc, char **argv) {
  cl = new CL;
  CHECK(cl != nullptr);

  TestCL();

  delete cl;

  printf("OK\n");
  return 0;
}
