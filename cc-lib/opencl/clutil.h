
#ifndef _CC_LIB_OPENCL_CLUTIL_H
#define _CC_LIB_OPENCL_CLUTIL_H

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_platform.h>

// Maybe make this not depend on base/logging?
// We only need some basic asserts.
#include "base/logging.h"
#include "base/stringprintf.h"

// TODO: Some profiling support via clGetEventProfilingInfo.

// Boilerplate. There should probably just be one CL instance per program.
struct CL {
  // Better compatibility with CL.
  using uchar = uint8_t;

  explicit CL(int verbose = 1);
  ~CL();

  static const char *ErrorString(cl_int err);

  /*
  cl_command_queue NewCommandQueue(bool out_of_order = true) {
    cl_queue_properties props =
      out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
    return clCreateCommandQueueWithProperties(
        context, devices[0],
        &props,
        nullptr);
  }
  */

  std::pair<cl_program, std::map<std::string, cl_kernel>> BuildKernels(
      const std::string &kernel_src,
      const std::set<std::string> &function_names,
      int fn_verbose = 1);

  std::pair<cl_program, cl_kernel>
  BuildOneKernel(const std::string &kernel_src,
                 const std::string &function_name,
                 int fn_verbose = 1);

  // Read the program binary for the first device and return it if it
  // is text.
  static std::optional<std::string> DecodeProgram(cl_program p);

  // Read an unspecified set of information about the device, so
  // that it can be printed out for debugging.
  std::map<std::string, std::string> DeviceInfo();

  // TODO: Make private.
  cl_uint num_devices = 0;
  cl_device_id *devices = nullptr;
  cl_context context;
  cl_command_queue queue;
 private:
  int verbose = 0;
};


// TODO: make CHECK_SUCCESS(exp) << stuff; work!
#define CHECK_SUCCESS(e) do {                                           \
    const int ret = (e);                                                \
    CHECK(ret == CL_SUCCESS) <<                                         \
      StringPrintf(__FILE__ ":%d in %s:\n"                              \
                   "Not successful with code %d (%s).\n",               \
                   __LINE__, __func__,                                  \
                   ret, CL::ErrorString(ret));                          \
  } while (0)

// Template implementations follow.

// Shares with the host memory and we don't control when it gets
// copied. This is quite inefficient. XXX maybe retire this.
template<class T>
static cl_mem BufferFromVector(cl_context context, bool readonly,
                               std::vector<T> *v) {
  return clCreateBuffer(context,
                        (readonly ? CL_MEM_READ_ONLY : 0) |
                        CL_MEM_USE_HOST_PTR,
                        sizeof (T) * v->size(),
                        (void *) v->data(),
                        nullptr);
}

// PERF: These are blocking copies; applications may be able to get
// much better throughput by batching...

// Creates a new buffer on the GPU and copies the memory there. They
// do not alias. Note that the command queue is not flushed, so you
// should not touch the source memory until it is.
template<class T>
static cl_mem CopyMemoryToGPU(cl_context context, cl_command_queue queue,
                              const std::vector<T> &v, bool readonly = false) {
  // TODO: Seems reasonable to allow empty memories, but clCreateBuffer
  // doesn't agree!
  CHECK(!v.empty()) << "CopyMemoryToGPU doesn't work for empty vectors";
  cl_int create_error = 0;
  cl_mem buf = clCreateBuffer(context,
                              (readonly ? CL_MEM_READ_ONLY : 0),
                              sizeof (T) * v.size(),
                              nullptr,
                              &create_error);
  CHECK_SUCCESS(create_error);
  CHECK_SUCCESS(clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0,
                                     sizeof (T) * v.size(), v.data(), 0,
                                     nullptr, nullptr));
  CHECK(buf != 0);
  return buf;
}

// PERF: Consider CL_MEM_WRITE_ONLY. Are there performance advantages?
template<class T>
static cl_mem CreateUninitializedGPUMemory(cl_context context, size_t n_items) {
  CHECK(n_items > 0) << "Empty buffers not supported :(";
  cl_int create_error = 0;
  size_t size_bytes = sizeof (T) * n_items;
  // printf("Bytes: %zu\n", size_bytes);
  cl_mem buf =
    clCreateBuffer(context, 0, size_bytes, nullptr, &create_error);
  CHECK_SUCCESS(create_error);
  CHECK(buf != 0);

  /*
  size_t actual = 0;
  CHECK_SUCCESS(
    clGetMemObjectInfo(buf,
                       CL_MEM_SIZE,
                       sizeof (actual),
                       &actual,
                       nullptr));
  printf("Actual byte size: %zu\n", actual);
  */
  return buf;
}

template<class T>
static std::vector<T> CopyBufferFromGPU(cl_command_queue queue,
                                        cl_mem buf, int n) {
  CHECK(n > 0) << "Empty buffers not supported :(";

  /*
  size_t actual = 0;
  CHECK_SUCCESS(
    clGetMemObjectInfo(buf,
                       CL_MEM_SIZE,
                       sizeof (actual),
                       &actual,
                       nullptr));
  printf("Actual byte size: %zu\n", actual);
  */

  std::vector<T> vec;
  vec.resize(n);
  size_t size_bytes = sizeof (T) * n;
  // printf("Size bytes: %zu\n", size_bytes);
  CHECK_SUCCESS(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, size_bytes,
                                    vec.data(),
                                    // No wait-list or event.
                                    0, nullptr,
                                    nullptr));
  clFinish(queue);
  return vec;
}

// Assumes the vector already has the correct size.
template<class T>
static void CopyBufferFromGPUTo(cl_command_queue queue,
                                cl_mem buf, std::vector<T> *vec) {
  // This would yield an error. We could support empty buffers
  // by just succeeding, I guess?
  CHECK(!vec->empty());
  CHECK_SUCCESS(
      clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof (T) * vec->size(),
                          vec->data(),
                          // No wait-list or event.
                          0, nullptr,
                          nullptr));
  clFinish(queue);
}

template<class T>
static void CopyBufferToGPU(cl_command_queue queue,
                            const std::vector<T> &vec, cl_mem buf) {
  // printf("%lld bytes\n", (int64_t)(sizeof (T) * vec.size()));
  CHECK_SUCCESS(clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0,
                                     sizeof (T) * vec.size(), vec.data(), 0,
                                     nullptr, nullptr));
  clFinish(queue);
}

// Create a sub-buffer of the given buffer that represents the
// supplied span. The argument is treated as an array of elements of
// type T (e.g. float, to help you avoid bugs where you forget that
// cl_mem is just bytes); T needs to be explicitly supplied.
// The input buffer allegedly cannot be itself a sub-buffer. Caller
// owns the sub-buffer and must release it.
template<class T>
static cl_mem SliceGPUMemory(cl_mem values, int start_idx, int count) {
  // The output region, sized in bytes.
  cl_buffer_region create_info = {
    .origin = (size_t)start_idx * sizeof (T),
    .size = (size_t)count * sizeof (T),
  };

  cl_int create_sub_buffer_error = 0;
  cl_mem sub_values = clCreateSubBuffer(values,
                                        // flags. 0 should inherit.
                                        0,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        (const void *)&create_info,
                                        &create_sub_buffer_error);
  CHECK_SUCCESS(create_sub_buffer_error);
  CHECK(sub_values != 0);
  return sub_values;
}

// Write n zero elements (of type T) to the cl memory. Intended for
// numeric T.
template<class T>
requires std::integral<T> || std::floating_point<T>
static void ZeroGPUMemory(cl_command_queue queue, cl_mem buf, size_t n) {
  T zero = 0;
  CHECK_SUCCESS(
      clEnqueueFillBuffer(queue,
                          buf,
                          // pattern and its size in bytes
                          &zero, sizeof (T),
                          // offset and size to fill (in BYTES)
                          0, (size_t)(n * sizeof (T)),
                          // no wait list or event
                          0, nullptr, nullptr));
  clFinish(queue);
}

#endif
