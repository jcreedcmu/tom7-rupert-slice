// Some of this code was derived from polyhedra.cc and yocto libraries.
// See LICENSES.

// Expected constant defines:
// NUM_VERTICES, an int, giving the number of vertices in the polyhedron.
// NUM_TRIANGLES, an int, giving the number of triangles in its triangulation.
// PARAM_H, a double, giving the offset to add for each parameter.

#define ERRORS_PER_CONFIG 11

typedef uchar uint8_t;
typedef uint uint32_t;
typedef int int32_t;
typedef ulong uint64_t;
typedef long int64_t;

typedef double4 quat4;

struct frame3 {
  double3 x, y, z, o;
};

inline struct frame3 RotationFrame(quat4 q) {
  struct frame3 frame;
  frame.x = (double3)(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                      (q.x * q.y + q.z * q.w) * 2.0,
                      (q.z * q.x - q.y * q.w) * 2.0);

  frame.y = (double3)((q.x * q.y - q.z * q.w) * 2.0,
                      q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                      (q.y * q.z + q.x * q.w) * 2.0);

  frame.z = (double3)((q.z * q.x + q.y * q.w) * 2.0,
                      (q.y * q.z - q.x * q.w) * 2.0,
                      q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z);

  frame.o = (double3)(0.0, 0.0, 0.0);

  return frame;
}

// PERF: Vectorize?
inline double2 RotateAndProjectPoint(struct frame3 f, double3 pt) {
  double3 ppt = f.x * pt.x + f.y * pt.y + f.z * pt.z + f.o;
  return ppt.xy;
}

inline double QuatDot(quat4 a, quat4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline double QuatLength(quat4 a) {
  return sqrt(dot(a, a));
}

inline quat4 QuatNormalize(quat4 q) {
  double l = QuatLength(q);
  return (l != 0.0) ? q / l : q;
}

double TriangleSignedDistance(double2 p0, double2 p1, double2 p2, double2 p) {
  // This function only:
  // Derived from code by Inigo Quilez; MIT license. See LICENSES.

  double2 e0 = p1 - p0;
  double2 e1 = p2 - p1;
  double2 e2 = p0 - p2;

  double2 v0 = p - p0;
  double2 v1 = p - p1;
  double2 v2 = p - p2;

  double2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  double2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  double2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);

  double s = e0.x * e2.y - e0.y * e2.x;
  double2 d =
    min(min((double2)(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
            (double2)(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
        (double2)(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));

  return -sqrt(d.x) * sign(d.y);
}


__kernel void RotateAndProject(// num_quats * 4
                               __global const double *restrict quats,
                               // Just one of these, NUM_VERTICES * 3 in size.
                               __global const double *restrict vertices_in,
                               // num_quats * NUM_VERTICES * 2
                               __global double *restrict vertices_out) {

  const int idx = get_global_id(0);

  const int quat_start = idx * 4;
  double4 quat = (double4)(quats[quat_start + 0],
                           quats[quat_start + 1],
                           quats[quat_start + 2],
                           quats[quat_start + 3]);

  struct frame3 frame = RotationFrame(quat);

  for (int i = 0; i < NUM_VERTICES; i++) {
    // Just one polyhedron.
    const int vertex_start = i * 3;
    double3 pt = (double3)(vertices_in[vertex_start + 0],
                           vertices_in[vertex_start + 1],
                           vertices_in[vertex_start + 2]);
    double2 ppt = RotateAndProjectPoint(frame, pt);
    const int out_start = (idx * 2 * NUM_VERTICES) + i * 2;
    vertices_out[out_start + 0] = ppt.x;
    vertices_out[out_start + 1] = ppt.y;
  }
}

__kernel void TweakQuaternions(// size num_quats * 4
                               __global double *restrict quats) {

  const int config_idx = get_global_id(0);

  const int config_start = config_idx * 4 * 5;

  quat4 q = (double4)(quats[config_start + (4 * 0) + 0],
                      quats[config_start + (4 * 0) + 1],
                      quats[config_start + (4 * 0) + 2],
                      quats[config_start + (4 * 0) + 3]);

  quat4 qa = q;
  qa.x += PARAM_H;
  qa = QuatNormalize(qa);
  quats[config_start + (4 * 1) + 0] = qa.x;
  quats[config_start + (4 * 1) + 1] = qa.y;
  quats[config_start + (4 * 1) + 2] = qa.z;
  quats[config_start + (4 * 1) + 3] = qa.w;

  quat4 qb = q;
  qb.y += PARAM_H;
  qb = QuatNormalize(qb);
  quats[config_start + (4 * 2) + 0] = qb.x;
  quats[config_start + (4 * 2) + 1] = qb.y;
  quats[config_start + (4 * 2) + 2] = qb.z;
  quats[config_start + (4 * 2) + 3] = qb.w;

  quat4 qc = q;
  qc.z += PARAM_H;
  qc = QuatNormalize(qc);
  quats[config_start + (4 * 3) + 0] = qc.x;
  quats[config_start + (4 * 3) + 1] = qc.y;
  quats[config_start + (4 * 3) + 2] = qc.z;
  quats[config_start + (4 * 3) + 3] = qc.w;

  quat4 qd = q;
  qd.w += PARAM_H;
  qd = QuatNormalize(qd);
  quats[config_start + (4 * 4) + 0] = qd.x;
  quats[config_start + (4 * 4) + 1] = qd.y;
  quats[config_start + (4 * 4) + 2] = qd.z;
  quats[config_start + (4 * 4) + 3] = qd.w;
}

__kernel void TweakTranslations(// size width * 2 * 3
                                __global double *restrict translate) {

  const int config_idx = get_global_id(0);

  const int config_start = config_idx * 2 * 3;

  const double tx = translate[config_start + 0];
  const double ty = translate[config_start + 1];

  // (x+h, y)
  translate[config_start + 2] = tx + PARAM_H;
  translate[config_start + 3] = ty;
  // (x, y+h)
  translate[config_start + 4] = tx;
  translate[config_start + 5] = ty + PARAM_H;
}


__kernel void ComputeError(// One triangulation shared among all the
                           // configurations. NUM_TRIANGLES * 3
                           __constant int *restrict triangles,
                           // The main configuration, and one tweaked
                           // version for each parameter.
                           // NUM_VERTICES * 2 * 5
                           __global const double *restrict outer_vertices,
                           // NUM_VERTICES * 2 * 5
                           __global const double *restrict inner_vertices,
                           // translations to apply to inner vertices.
                           // we have three per configuration; one main
                           // translation and a perturbed version for x
                           // and y.
                           __global const double *restrict translate,
                           // The main configuration's error, and one
                           // error for each tweaked combination.
                           // width * 11
                           __global double *restrict error_values) {
  // The error idx denotes a pair of 2D meshes; we'll compute a single
  // error value and write it into the corresponding slot in error_values.
  const int error_idx = get_global_id(0);

  // The index of the main configuration. This gives us the starting
  // position for the 5 quaternions in outer_vertices and inner_vertices.
  const int config_idx = error_idx / 11;
  const int config_offset = error_idx % 11;

  //          0   1  2  3  4
  // outer: main  x  y  z  w
  // inner: main' x' y' z' w'
  //              5  6  7  8
  //
  // translation: main'' x'' y''
  //                     9   10
  //
  // Index 0 is (main, main', main''); note we only compute this one time.
  // 1 is (x, main', main''). 2 is (y, main', main''), ...
  // 5 is (main, x', main''). 6 is (main, y', main''), ...
  // 9 is (main, main', x''), 10 is (main, main', y'')

  const int outer_tweak_idx =
    (config_offset >= 1 && config_offset <= 4) ? config_offset : 0;
  const int inner_tweak_idx =
    (config_offset >= 5 && config_offset <= 8) ? config_offset - 4 : 0;
  const int translate_tweak_idx =
    (config_offset >= 9 && config_offset <= 10) ? config_offset - 8 : 0;

  // The outer vertices we're computing error for (size num_vertices * 2)
  const double *overt = outer_vertices +
    ((config_idx * 5) + outer_tweak_idx) * NUM_VERTICES * 2;

  // The inner vertices we're computing error for (size num_vertices * 2)
  const double *ivert = inner_vertices +
    ((config_idx * 5) + inner_tweak_idx) * NUM_VERTICES * 2;

  const double *tr = translate +
    ((config_idx * 3) + translate_tweak_idx) * 2;
  const double2 trans = (double2)(tr[0], tr[1]);

  // The total error is just a sum over all vertices:
  double total_error = 0.0;
  for (int inner_idx = 0; inner_idx < NUM_VERTICES; inner_idx++) {
    double2 pt = (double2)(ivert[inner_idx * 2 + 0], ivert[inner_idx * 2 + 1]);
    pt += trans;

    // The error for the point is the minimum rectified distance to any
    // triangle in the outer mesh.
    double point_error = 1.0e30;
    for (int tidx = 0; tidx < NUM_TRIANGLES; tidx++) {
      int va = triangles[tidx * 3 + 0];
      int vb = triangles[tidx * 3 + 1];
      int vc = triangles[tidx * 3 + 2];
      double2 a = (double2)(overt[va * 2 + 0], overt[va * 2 + 1]);
      double2 b = (double2)(overt[vb * 2 + 0], overt[vb * 2 + 1]);
      double2 c = (double2)(overt[vc * 2 + 0], overt[vc * 2 + 1]);

      double terr = TriangleSignedDistance(a, b, c, pt);

      // We can end early if we are completely within a triangle, but
      // this may actually be harmful on the GPU. Currently this code
      // is all branchless.
      point_error = min(point_error, max(terr, 0.0));
    }

    total_error += point_error;
  }

  error_values[error_idx] = total_error;
}

__kernel void CheckForSolution(__global const double* restrict error_values,
                               __global int32_t* restrict solution_index) {
  const int32_t config_idx = get_global_id(0);
  const double error = error_values[config_idx * ERRORS_PER_CONFIG];

  if (error == 0.0) {
    atomic_min(solution_index, config_idx);
  }
}

const double LEARNING_RATE = 0.01;
// Using the errors, estimate the gradient numerically for each of the
// ten parameters. Use this to compute a new sample value, and write
// that over the first quaternion in each group. (The perturbed versions
// are computed by TweakQuaternions.)
__kernel void GradientDescent(// The main configuration's quaternion,
                              // followed by four versions with a single
                              // perturbed parameter (x, y, z, w).
                              // width * 4 * 5
                              __global double *restrict outer_quats,
                              // width * 4 * 5
                              __global double *restrict inner_quats,
                              // width * 2 * 3
                              __global double *restrict translate,
                              // The main configuration's error, and one
                              // error for each tweaked combination.
                              // width * ERRORS_PER_CONFIG
                              __global const double *restrict error_values) {

  // The config idx denotes a pair of outer and inner quaternions
  // (from the parallel arrays), giving the rotation. Each one is
  // followed by four perturbed quaternions.

  const int config_idx = get_global_id(0);

  double *oquats = outer_quats + config_idx * 5 * 4;
  double *iquats = inner_quats + config_idx * 5 * 4;
  double *ts = translate + config_idx * 3 * 2;

  const double *errs = error_values + config_idx * ERRORS_PER_CONFIG;

  const quat4 oq = (double4)(oquats[0], oquats[1], oquats[2], oquats[3]);
  const quat4 iq = (double4)(iquats[0], iquats[1], iquats[2], iquats[3]);
  const double2 tx = (double2)(ts[0], ts[1]);

  const double main_err = errs[0];
  double4 oq_grad = (double4)((errs[1] - main_err) / PARAM_H,
                              (errs[2] - main_err) / PARAM_H,
                              (errs[3] - main_err) / PARAM_H,
                              (errs[4] - main_err) / PARAM_H);
  double4 iq_grad = (double4)((errs[5] - main_err) / PARAM_H,
                              (errs[6] - main_err) / PARAM_H,
                              (errs[7] - main_err) / PARAM_H,
                              (errs[8] - main_err) / PARAM_H);

  double2 tx_grad = (double2)((errs[9] - main_err) / PARAM_H,
                              (errs[10] - main_err) / PARAM_H);

  quat4 new_oq = QuatNormalize(oq - LEARNING_RATE * oq_grad);
  quat4 new_iq = QuatNormalize(iq - LEARNING_RATE * iq_grad);
  double2 new_tx = tx - LEARNING_RATE * tx_grad;

  oquats[0] = new_oq.x;
  oquats[1] = new_oq.y;
  oquats[2] = new_oq.z;
  oquats[3] = new_oq.w;

  iquats[0] = new_iq.x;
  iquats[1] = new_iq.y;
  iquats[2] = new_iq.z;
  iquats[3] = new_iq.w;

  ts[0] = new_tx.x;
  ts[1] = new_tx.y;
}

