#include "clutil.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <cstdint>
#include <utility>
#include <vector>
#include <map>
#include <set>

#include <CL/cl.h>
#include <CL/cl_platform.h>

#include "timer.h"
#include "base/logging.h"
#include "base/stringprintf.h"

using namespace std;

using uint8 = uint8_t;
using uint64 = uint64_t;

CL::CL(int verbose) : verbose(verbose) {
  cl_uint num_platforms;
  cl_platform_id platform = nullptr;
  CHECK(CL_SUCCESS == clGetPlatformIDs(0, nullptr, &num_platforms));
  if (verbose > 0) {
    fprintf(stderr, "CL:: Number of platforms: %d.\n", num_platforms);
  }

  // Choose the first platform that has a GPU.
  int chosen_platform_id = -1;
  if (num_platforms > 0) {
    cl_platform_id *platforms =
      (cl_platform_id *)malloc(num_platforms * sizeof (cl_platform_id));
    CHECK(CL_SUCCESS == clGetPlatformIDs(num_platforms, platforms, nullptr));
    for (int i = 0; i < num_platforms; i++) {
      struct Prop {
        cl_platform_info key;
        size_t size;
        char *value;
      };
      Prop props[] = {
        {CL_PLATFORM_PROFILE, 0, nullptr},
        {CL_PLATFORM_VERSION, 0, nullptr},
        {CL_PLATFORM_NAME, 0, nullptr},
        {CL_PLATFORM_VENDOR, 0, nullptr},
        {CL_PLATFORM_EXTENSIONS, 0, nullptr},
      };
      for (Prop &prop : props) {
        clGetPlatformInfo(
            platforms[i], prop.key, 0, nullptr, &prop.size);
        prop.value = (char *)malloc(prop.size + 1);
        clGetPlatformInfo(
            platforms[i], prop.key, prop.size, prop.value, nullptr);
      }

      if (verbose > 1) {
        fprintf(stderr,
                "% 4d. %s (%s):\n"
                "      %s; %s\n"
                // ?
                "      %s\n",
                i, props[2].value, props[0].value,
                props[1].value, props[3].value, props[4].value);
      }

      cl_uint platform_devices = 0;
      if (CL_SUCCESS == clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                                       0, nullptr, &platform_devices)) {
        if (verbose > 0) {
          fprintf(stderr,
                  "      Number of GPUs: %d\n", (int)platform_devices);
        }
        if (chosen_platform_id == -1)
          chosen_platform_id = i;
      } else {
        if (verbose > 0) {
          fprintf(stderr,
                  "      NO GPU devices.\n");
        }
      }

      for (Prop &prop : props) free(prop.value);
    }

    if (chosen_platform_id == -1) {
      LOG(FATAL) << "There were no platforms with GPUs.";
    }

    if (verbose > 1) {
      fprintf(stderr, "Using platform %d.\n", chosen_platform_id);
    }
    platform = platforms[chosen_platform_id];
    free(platforms);
  }

  // Get the GPU device.
  CHECK(CL_SUCCESS == clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                                     0, nullptr, &num_devices));
  CHECK(num_devices > 0) << "Platform should only be selected if it "
    "reports having devices??";

  devices = (cl_device_id *)malloc(num_devices * sizeof (cl_device_id));
  CHECK(CL_SUCCESS == clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                                     num_devices, devices, nullptr));

  context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, nullptr);


  constexpr bool out_of_order = false;
  // TODO: Try device queue?
  const cl_queue_properties props =
    out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
  queue = clCreateCommandQueueWithProperties(
      context, devices[0],
      &props,
      nullptr);
}

CL::~CL() {
  if (verbose > 0) {
    fprintf(stderr, "Destroying CL.\n");
  }
  CHECK_SUCCESS(clReleaseCommandQueue(queue));
  CHECK_SUCCESS(clReleaseContext(context));
  free(devices);
  if (verbose > 0) {
    fprintf(stderr, "CL destroyed.\n");
  }
}


std::optional<std::string> CL::DecodeProgram(cl_program p) {
  cl_uint number_of_devices;
  CHECK_SUCCESS(clGetProgramInfo(p, CL_PROGRAM_NUM_DEVICES,
                                 sizeof (cl_uint),
                                 &number_of_devices,
                                 nullptr));

  std::vector<size_t> sizes(number_of_devices, 0);
  CHECK_SUCCESS(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES,
                                 number_of_devices * sizeof(size_t),
                                 sizes.data(),
                                 nullptr));

  std::vector<char *> binaries(number_of_devices, nullptr);
  for (int i = 0; i < binaries.size(); i++) {
    binaries[i] = (char *)malloc(sizes[i] + 1);
  }

  CHECK_SUCCESS(clGetProgramInfo(p, CL_PROGRAM_BINARIES,
                                 number_of_devices * sizeof(char*),
                                 binaries.data(),
                                 nullptr));

  std::vector<string> sbinaries;
  sbinaries.reserve(binaries.size());
  for (int i = 0; i < binaries.size(); i++) {
    sbinaries.push_back(std::string(binaries[i], sizes[i]));
    free(binaries[i]);
  }
  binaries.clear();

  printf("There were %d binaries, sizes ", (int)number_of_devices);
  for (size_t i : sizes) printf("%d, ", (int)i);

  // Assume binary 0 is nvidia.
  // XXX We should try to detect this case more robustly!
  if (!sbinaries.empty()) return {sbinaries[0]};
  return {};
}


// TODO: PERF: Option to enable/disable fast-math style optimizations:
// khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clBuildProgram.html
// which should be suitable for learning applications.
std::pair<cl_program, std::map<std::string, cl_kernel>> CL::BuildKernels(
    const string &kernel_src,
    const std::set<string> &function_names,
    int arg_verbose) {
  int fn_verbose = std::max(verbose, arg_verbose);
  Timer gpu_compile;
  const char *sources[] = { kernel_src.c_str() };
  size_t source_size[] = { kernel_src.size() };
  cl_program program =
    clCreateProgramWithSource(context, 1, sources, source_size, nullptr);
  if (CL_SUCCESS !=
      clBuildProgram(program, 1, devices,
                     // command-line options.
                     nullptr,
                     nullptr, nullptr)) {
    size_t blsize;

    CHECK(CL_SUCCESS ==
          clGetProgramBuildInfo(program, devices[0],
                                CL_PROGRAM_BUILD_LOG, 0, nullptr, &blsize));
    char *build_log = (char *)malloc(blsize + 1);
    CHECK(CL_SUCCESS ==
          clGetProgramBuildInfo(program, devices[0],
                                CL_PROGRAM_BUILD_LOG, blsize, build_log,
                                nullptr));
    build_log[blsize] = 0;
    fprintf(stderr, "Failed to compile:\n %s", build_log);
    free(build_log);
    LOG(FATAL) << "Kernel compilation failed.";
  }

  std::map<string, cl_kernel> kernels;

  for (const string &function_name : function_names) {
    cl_int kernel_error = 0;
    cl_kernel kernel = clCreateKernel(program, function_name.c_str(),
                                      &kernel_error);
    CHECK_SUCCESS(kernel_error);
    if (fn_verbose > 0) {
      fprintf(stderr, "Compiled %s in %.1fms.\n",
              function_name.c_str(), gpu_compile.MS());
    }
    kernels[function_name] = kernel;
  }

  return make_pair(program, kernels);
}

pair<cl_program, cl_kernel> CL::BuildOneKernel(const string &kernel_src,
                                               const string &function_name,
                                               int arg_verbose) {
  const int fn_verbose = std::max(verbose, arg_verbose);
  const auto &[prog, kernels] =
    CL::BuildKernels(kernel_src, std::set<string>({function_name}), fn_verbose);

  auto it = kernels.find(function_name);
  CHECK(it != kernels.end());
  return make_pair(prog, it->second);
}

// Read an unspecified set of information about the device, so
// that it can be printed out for debugging.
std::map<std::string, std::string> CL::DeviceInfo() {
  if (devices == 0) return {{"(error)", "(no devices)"}};

  std::map<std::string, std::string> ret;

  {
    cl_ulong local_mem_size = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE,
                              sizeof(local_mem_size), &local_mem_size, nullptr);
    if (err == CL_SUCCESS) {
      ret["local mem size"] =
        StringPrintf("%lld kb", (int64_t)local_mem_size);
    } else {
      ret["local mem size"] = "(error)";
    }
  }

  {
    cl_ulong global_mem_size = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof(global_mem_size), &global_mem_size,
                              nullptr);
    if (err == CL_SUCCESS) {
      ret["global mem size"] =
        StringPrintf("%lld MB", (int64_t)(global_mem_size / (1024 * 1024)));
    } else {
      ret["global mem size"] = "(error)";
    }
  }

  {
    cl_uint max_compute_units = 0;
    int err =
        clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(max_compute_units), &max_compute_units, nullptr);
    if (err == CL_SUCCESS) {
      ret["max compute units"] = StringPrintf("%d", max_compute_units);
    } else {
      ret["max compute units"] = "(error)";
    }
  }

  {
    size_t max_work_group_size = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof(max_work_group_size), &max_work_group_size,
                              nullptr);
    if (err == CL_SUCCESS) {
      ret["max work group size"] =
          StringPrintf("%lld", (int64_t)max_work_group_size);
    } else {
      ret["max work group size"] = "(error)";
    }
  }

  {
    cl_uint max_clock_frequency = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                              sizeof(max_clock_frequency), &max_clock_frequency,
                              nullptr);
    if (err == CL_SUCCESS) {
      ret["max clock frequency"] = StringPrintf("%d MHz", max_clock_frequency);
    } else {
      ret["max clock frequency"] = "(error)";
    }
  }

  {
    char device_name[1024];
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(device_name),
                          device_name, nullptr);
    if (err == CL_SUCCESS) {
      ret["device name"] = device_name;
    } else {
      ret["device name"] = "(error)";
    }
  }

  {
    char device_vendor[1024];
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR,
                              sizeof(device_vendor), device_vendor, nullptr);
    if (err == CL_SUCCESS) {
      ret["device vendor"] = device_vendor;
    } else {
      ret["device vendor"] = "(error)";
    }
  }

  {
    char device_version[1024];
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_VERSION,
                              sizeof(device_version), device_version, nullptr);
    if (err == CL_SUCCESS) {
      ret["device version"] = device_version;
    } else {
      ret["device version"] = "(error)";
    }
  }

  {
    char driver_version[1024];
    int err = clGetDeviceInfo(devices[0], CL_DRIVER_VERSION,
                              sizeof(driver_version), driver_version, nullptr);
    if (err == CL_SUCCESS) {
      ret["driver version"] = driver_version;
    } else {
      ret["driver version"] = "(error)";
    }
  }

  {
    cl_ulong max_mem_alloc_size = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                              sizeof(max_mem_alloc_size), &max_mem_alloc_size,
                              nullptr);
    if (err == CL_SUCCESS) {
      ret["max mem alloc size"] = StringPrintf(
          "%lld MB", (int64_t)(max_mem_alloc_size / (1024 * 1024)));
    } else {
      ret["max mem alloc size"] = "(error)";
    }
  }

  {
    cl_ulong max_constant_buffer_size = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                              sizeof(max_constant_buffer_size),
                              &max_constant_buffer_size, nullptr);
    if (err == CL_SUCCESS) {
      ret["max constant buffer size"] =
          StringPrintf("%lld KB", (int64_t)(max_constant_buffer_size / 1024));
    } else {
      ret["max constant buffer size"] = "(error)";
    }
  }

  {
    cl_uint max_samplers = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_SAMPLERS,
                              sizeof(max_samplers), &max_samplers, nullptr);
    if (err == CL_SUCCESS) {
      ret["max samplers"] = StringPrintf("%d", max_samplers);
    } else {
      ret["max samplers"] = "(error)";
    }
  }

  {
    cl_uint max_read_image_args = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_READ_IMAGE_ARGS,
                              sizeof(max_read_image_args), &max_read_image_args,
                              nullptr);
    if (err == CL_SUCCESS) {
      ret["max read image args"] = StringPrintf("%d", max_read_image_args);
    } else {
      ret["max read image args"] = "(error)";
    }
  }

  {
    cl_uint max_write_image_args = 0;
    int err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                              sizeof(max_write_image_args),
                              &max_write_image_args, nullptr);
    if (err == CL_SUCCESS) {
      ret["max write image args"] = StringPrintf("%d", max_write_image_args);
    } else {
      ret["max write image args"] = "(error)";
    }
  }

  // More here...
  return ret;
}


// static
const char *CL::ErrorString(cl_int err) {
  switch (err) {
#define MAKE_ERROR(e) case e: return #e ;
  MAKE_ERROR(CL_SUCCESS);
  MAKE_ERROR(CL_DEVICE_NOT_FOUND);
  MAKE_ERROR(CL_DEVICE_NOT_AVAILABLE);
  MAKE_ERROR(CL_COMPILER_NOT_AVAILABLE);
  MAKE_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
  MAKE_ERROR(CL_OUT_OF_RESOURCES);
  MAKE_ERROR(CL_OUT_OF_HOST_MEMORY);
  MAKE_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
  MAKE_ERROR(CL_MEM_COPY_OVERLAP);
  MAKE_ERROR(CL_IMAGE_FORMAT_MISMATCH);
  MAKE_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
  MAKE_ERROR(CL_BUILD_PROGRAM_FAILURE);
  MAKE_ERROR(CL_MAP_FAILURE);
  MAKE_ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
  MAKE_ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
  MAKE_ERROR(CL_COMPILE_PROGRAM_FAILURE);
  MAKE_ERROR(CL_LINKER_NOT_AVAILABLE);
  MAKE_ERROR(CL_LINK_PROGRAM_FAILURE);
  MAKE_ERROR(CL_DEVICE_PARTITION_FAILED);
  MAKE_ERROR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
  MAKE_ERROR(CL_INVALID_VALUE);
  MAKE_ERROR(CL_INVALID_DEVICE_TYPE);
  MAKE_ERROR(CL_INVALID_PLATFORM);
  MAKE_ERROR(CL_INVALID_DEVICE);
  MAKE_ERROR(CL_INVALID_CONTEXT);
  MAKE_ERROR(CL_INVALID_QUEUE_PROPERTIES);
  MAKE_ERROR(CL_INVALID_COMMAND_QUEUE);
  MAKE_ERROR(CL_INVALID_HOST_PTR);
  MAKE_ERROR(CL_INVALID_MEM_OBJECT);
  MAKE_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
  MAKE_ERROR(CL_INVALID_IMAGE_SIZE);
  MAKE_ERROR(CL_INVALID_SAMPLER);
  MAKE_ERROR(CL_INVALID_BINARY);
  MAKE_ERROR(CL_INVALID_BUILD_OPTIONS);
  MAKE_ERROR(CL_INVALID_PROGRAM);
  MAKE_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
  MAKE_ERROR(CL_INVALID_KERNEL_NAME);
  MAKE_ERROR(CL_INVALID_KERNEL_DEFINITION);
  MAKE_ERROR(CL_INVALID_KERNEL);
  MAKE_ERROR(CL_INVALID_ARG_INDEX);
  MAKE_ERROR(CL_INVALID_ARG_VALUE);
  MAKE_ERROR(CL_INVALID_ARG_SIZE);
  MAKE_ERROR(CL_INVALID_KERNEL_ARGS);
  MAKE_ERROR(CL_INVALID_WORK_DIMENSION);
  MAKE_ERROR(CL_INVALID_WORK_GROUP_SIZE);
  MAKE_ERROR(CL_INVALID_WORK_ITEM_SIZE);
  MAKE_ERROR(CL_INVALID_GLOBAL_OFFSET);
  MAKE_ERROR(CL_INVALID_EVENT_WAIT_LIST);
  MAKE_ERROR(CL_INVALID_EVENT);
  MAKE_ERROR(CL_INVALID_OPERATION);
  MAKE_ERROR(CL_INVALID_GL_OBJECT);
  MAKE_ERROR(CL_INVALID_BUFFER_SIZE);
  MAKE_ERROR(CL_INVALID_MIP_LEVEL);
  MAKE_ERROR(CL_INVALID_GLOBAL_WORK_SIZE);
  MAKE_ERROR(CL_INVALID_PROPERTY);
  MAKE_ERROR(CL_INVALID_IMAGE_DESCRIPTOR);
  MAKE_ERROR(CL_INVALID_COMPILER_OPTIONS);
  MAKE_ERROR(CL_INVALID_LINKER_OPTIONS);
  MAKE_ERROR(CL_INVALID_DEVICE_PARTITION_COUNT);
  default: return "??";
  }
}
