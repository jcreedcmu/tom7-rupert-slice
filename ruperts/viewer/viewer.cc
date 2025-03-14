
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstdint>
#include <unistd.h>

#include "SDL.h"
#include "SDL_error.h"
#include "SDL_events.h"
#include "SDL_joystick.h"
#include "SDL_keyboard.h"
#include "SDL_keysym.h"
#include "SDL_main.h"
#include "SDL_mouse.h"
#include "SDL_timer.h"
#include "SDL_video.h"
#include "sdl/sdlutil.h"
#include "sdl/font.h"
#include "sdl/cursor.h"
#include "sdl/chars.h"

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "image.h"
#include "lines.h"
#include "periodically.h"
#include "randutil.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"

#include "mesh.h"
#include "yocto_matht.h"

#include "mov.h"
#include "mov-recorder.h"

using namespace std;

using int64 = int64_t;
using uint32 = uint32_t;

static constexpr bool TRACE = false;

static constexpr const char *FONT_PNG = "../../cc-lib/sdl/font.png";
static Font *font = nullptr, *font2x = nullptr, *font4x = nullptr;

static SDL_Cursor *cursor_arrow = nullptr, *cursor_bucket = nullptr;
static SDL_Cursor *cursor_hand = nullptr, *cursor_hand_closed = nullptr;
static SDL_Cursor *cursor_eraser = nullptr;

#define SCREENW 1024
#define SCREENH 1024

static SDL_Joystick *joystick = nullptr;
static SDL_Surface *screen = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using frame3 = yocto::frame<double, 3>;
using mat3 = yocto::mat<double, 3>;
using mat4 = yocto::mat<double, 4>;

TriangularMesh3D TransformMesh(const frame3 &frame,
                               const TriangularMesh3D &mesh) {
  TriangularMesh3D out = mesh;
  for (vec3 &v : out.vertices) v = yocto::transform_point(frame, v);
  return out;
}

TriangularMesh3D TransformMesh(const mat4 &mtx,
                               const TriangularMesh3D &mesh) {
  TriangularMesh3D out = mesh;
  for (vec3 &v : out.vertices) v = yocto::transform_point(mtx, v);
  return out;
}

inline vec3 ProjectPt(const mat4 &proj, const vec3 &point) {
  vec4 pp = proj * vec4{.x = point.x, .y = point.y, .z = point.z, .w = 1.0};
  return vec3{.x = pp.x / pp.w, .y = pp.y / pp.w, .z = pp.z / pp.w};
}

TriangularMesh3D ProjectMesh(const mat4 &proj,
                             const TriangularMesh3D &mesh) {
  TriangularMesh3D out = mesh;
  for (vec3 &v : out.vertices) v = ProjectPt(proj, v);
  return out;
}

enum class Speed {
  PLAY,
  PAUSE,
};

enum class View {
  ORTHOGRAPHIC,
  PERSPECTIVE,
};

namespace {
template<class F>
struct ScopeExit {
  ScopeExit(F &&f) : f(std::forward<F>(f)) {}
  ~ScopeExit() { f(); }
  F f;
};
}

// Gamepad values, from NES!
#define INPUT_R (1<<7)
#define INPUT_L (1<<6)
#define INPUT_D (1<<5)
#define INPUT_U (1<<4)
#define INPUT_T (1<<3)
#define INPUT_S (1<<2)
#define INPUT_B (1<<1)
#define INPUT_A (1   )

struct Scene {
  vec3 camera_pos = vec3{-1.0, -.03, 4.0};
  const vec3 object_pos = vec3{0.0, 0.0, 0.0};

  void Reset() {
    camera_pos = vec3{-1.0, -.03, 4.0};
  }

  explicit Scene(TriangularMesh3D mesh) : original_mesh(std::move(mesh)) {
    OrientMesh(&original_mesh);
    Reset();
  }

  void Draw(ImageRGBA *img) {
    img->Clear32(0x000000FF);

    constexpr double SCALE = 300.0;

    // Ideally this should happen from the transform itself
    auto ToScreen = [img](vec3 v) {
        // 0,0 in center.
        int x = std::round((img->Width() >> 1) + v.x * SCALE);
        int y = std::round((img->Height() >> 1) - v.y * SCALE);
        return std::make_pair(x, y);
      };

    auto DrawLine = [&](const vec3 &a, const vec3 &b,
                        uint32_t color) {
        const auto &[x0, y0] = ToScreen(a);
        const auto &[x1, y1] = ToScreen(b);
        auto clip = ClipLineToRectangle<float>(x0, y0, x1, y1,
                                               0, 0, SCREENW, SCREENH);
        if (clip.has_value()) {
          const auto &[cx0, cy0, cx1, cy1] = clip.value();
          img->BlendLine32(cx0, cy0, cx1, cy1, color);
        }
      };

    constexpr double FOVY = 1.0; // 1 radian is about 60 deg
    constexpr double ASPECT_RATIO = 1.0;
    constexpr double NEAR_PLANE = 0.1;
    constexpr double FAR_PLANE = 1000.0;
    mat4 persp = yocto::perspective_mat(FOVY, ASPECT_RATIO, NEAR_PLANE,
                                        FAR_PLANE);

    // frame3 frame = translation_frame(-camera_pos);

    frame3 frame = inverse(lookat_frame(camera_pos, object_pos,
                                        vec3{0.0, 1.0, 0.0}));

    TriangularMesh3D mesh = TransformMesh(persp,
                                          TransformMesh(frame, original_mesh));

    vec3 xpos = ProjectPt(persp, transform_point(frame, vec3{1, 0, 0}));
    vec3 ypos = ProjectPt(persp, transform_point(frame, vec3{0, 1, 0}));
    vec3 zpos = ProjectPt(persp, transform_point(frame, vec3{0, 0, 1}));
    vec3 origin = ProjectPt(persp, transform_point(frame, vec3{0, 0, 0}));

    for (const auto &[a, b, c] : mesh.triangles) {
      vec3 v0 = mesh.vertices[a];
      vec3 v1 = mesh.vertices[b];
      vec3 v2 = mesh.vertices[c];

      bool clipped =
        (v0.z < 0 || v1.z < 0 || v2.z < 0) ||
        (v0.z > 1 || v1.z > 1 || v2.z > 1);

      vec3 ctr = (v0 + v1 + v2) / 3.0;
      vec3 normal = normalize(cross(v1 - v0, v2 - v0)) * 0.25;
      bool backface = normal.z > 0;

      uint32_t color = 0xFFFFFFAA;

      if (clipped) {
        color = 0x88000088;
      } else if (backface) {
        color = 0xFFFF0088;
      }

      DrawLine(v0, v1, color);
      DrawLine(v1, v2, color);
      DrawLine(v2, v0, color);
    }
  }

  TriangularMesh3D original_mesh;

};

struct UI {
  Scene scene;
  Speed speed = Speed::PLAY;
  View view = View::ORTHOGRAPHIC;
  uint8_t current_gamepad = 0;
  int64_t frames_drawn = 0;
  uint8_t last_jhat = 0;
  vec3 vel = vec3{0, 0, 0};

  UI(TriangularMesh3D mesh);
  void Loop();
  void Draw();
  void DrawGrid();

  void PlayPause();

  Periodically fps_per;

  enum class EventResult { NONE, DIRTY, EXIT, };
  EventResult HandleEvents();

  std::unique_ptr<ImageRGBA> drawing;
  int mousex = 0, mousey = 0;
  bool dragging = false;

  std::pair<int, int> drag_source = {-1, -1};
  int drag_handlex = 0, drag_handley = 0;

  std::unique_ptr<MovRecorder> mov;
};

UI::UI(TriangularMesh3D mesh) : scene(std::move(mesh)), fps_per(1.0 / 60.0) {
  drawing.reset(new ImageRGBA(SCREENW, SCREENH));
  CHECK(drawing != nullptr);
  drawing->Clear32(0x000000FF);
}

void UI::PlayPause() {
  if (speed == Speed::PAUSE) {
    speed = Speed::PLAY;
  } else {
    speed = Speed::PAUSE;
  }
}

UI::EventResult UI::HandleEvents() {
  bool ui_dirty = false;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_QUIT:
      printf("QUIT.\n");
      return EventResult::EXIT;

    case SDL_MOUSEMOTION: {
      SDL_MouseMotionEvent *e = (SDL_MouseMotionEvent*)&event;

      [[maybe_unused]] const int oldx = mousex, oldy = mousey;

      mousex = e->x;
      mousey = e->y;

      #if 0
      if (dragging) {
        switch (mode) {
        case Mode::PLAY:

          break;
        default:;
        }
        ui_dirty = true;
      }
      #endif
      break;
    }

    case SDL_JOYAXISMOTION: {
      //   ^-            ^-
      //   1  <0>        3  <4>
      //   v+ - +        v+ - +

      auto MapAxis = [](int a) {
          bool neg = a < 0;
          if (neg) a = -a;

          // dead zone at top and bottom
          constexpr int lo = 8192;
          constexpr int hi = 1024;

          if (a < lo) {
            return 0.0;
          } else {
            double m = std::clamp((a - lo) / (32768.0 - lo - hi), 0.0, 1.0);
            return neg ? -m : m;
          }
        };

      SDL_JoyAxisEvent *j = (SDL_JoyAxisEvent *)&event;
      switch (j->axis) {
      case 0:
        // left and right
        vel.x = MapAxis(j->value);
        break;
      case 1:
        // forward and back
        vel.z = MapAxis(j->value);
        break;

      case 3:
      case 4:
        // TODO: Look
        break;
      }

      if (TRACE) {
        printf("%02x.%02x.%02x = %d\n",
               j->type, j->which, j->axis,
               (int)j->value);
      }
      break;
    }

    case SDL_KEYDOWN: {
      switch (event.key.keysym.sym) {
      case SDLK_ESCAPE:
        printf("ESCAPE.\n");
        return EventResult::EXIT;

      case SDLK_HOME: {
        // TODO: Reset pos

        speed = Speed::PAUSE;
        ui_dirty = true;
        break;
      }

      // Delete, backspace, ...?

      case SDLK_LEFT: {
        if (event.key.keysym.mod & KMOD_CTRL) {

        } else {

        }
        ui_dirty = true;
        break;
      }

      case SDLK_RIGHT: {
        if (event.key.keysym.mod & KMOD_CTRL) {

        } else {

        }
        ui_dirty = true;
        break;
      }

      case SDLK_UP: {
        if (event.key.keysym.mod & KMOD_CTRL) {

        }
        ui_dirty = true;
        break;
      }

      case SDLK_DOWN: {
        if (event.key.keysym.mod & KMOD_CTRL) {

        }
        ui_dirty = true;
        break;
      }

      case SDLK_SPACE: {
        PlayPause();
        ui_dirty = true;
        break;
      }

      case SDLK_KP_PLUS:
      case SDLK_EQUALS:
      case SDLK_PLUS:
        // TODO: Speed up
        ui_dirty = true;
        break;

      case SDLK_KP_MINUS:
      case SDLK_MINUS:
        // TODO: Speed down
        ui_dirty = true;
        break;

      case SDLK_RETURN:
      case SDLK_KP_ENTER:
        switch (view) {
        case View::ORTHOGRAPHIC: view = View::PERSPECTIVE; break;
        case View::PERSPECTIVE: view = View::ORTHOGRAPHIC; break;
        default: break;
        }
        ui_dirty = true;
        break;

      case SDLK_r: {
        if (mov.get() != nullptr) {
          // Note that this does pause while we finish output, which I
          // think is desirable. But we could make that also be
          // asynchronous.
          mov.reset();
          printf("Stopped recording.\n");
        } else {
          std::string filename = std::format("rec-{}.mov", time(nullptr));
          mov.reset(
              new MovRecorder(MOV::OpenOut(filename, SCREENW, SCREENH,
                                           MOV::DURATION_60,
                                           MOV::Codec::PNG_MINIZ)));
        }
      }

      default:;
      }
      break;
    }

    case SDL_MOUSEBUTTONDOWN: {
      // LMB/RMB, drag, etc.
      SDL_MouseButtonEvent *e = (SDL_MouseButtonEvent*)&event;
      mousex = e->x;
      mousey = e->y;

      dragging = true;

      break;
    }

    case SDL_MOUSEBUTTONUP: {
      // LMB/RMB, drag, etc.
      dragging = false;

      ui_dirty = true;
      drag_source = {-1, -1};
      // SDL_SetCursor(cursor_hand);
      break;
    }

    case SDL_JOYBUTTONDOWN:
      //    4              5
      //
      //
      //                   3
      //
      // 6    7        2       1
      //
      //                   0

      switch (event.jbutton.button) {
      case 2: ui_dirty = true; break;
      case 3: ui_dirty = true; break;
      case 6: current_gamepad |= INPUT_S; break;
      case 7: current_gamepad |= INPUT_T; break;
      case 0: current_gamepad |= INPUT_B; break;
      case 1: current_gamepad |= INPUT_A; break;
      case 4:
        ui_dirty = true;
        break;
      case 5:
        ui_dirty = true;
        break;

      case 8:
        // I think this is left analog (pressed)
        break;

      default:
        printf("Button %d unmapped.\n", event.jbutton.button);
      }
      break;

    case SDL_JOYBUTTONUP:
      switch (event.jbutton.button) {
      case 2: break;
      case 3: break;
      case 4: break;
      case 5: break;
      case 6: current_gamepad &= ~INPUT_S; break;
      case 7: current_gamepad &= ~INPUT_T; break;
      case 0: current_gamepad &= ~INPUT_B; break;
      case 1: current_gamepad &= ~INPUT_A; break;
      default:
        printf("Button %d unmapped.\n", event.jbutton.button);
      }
      break;

    case SDL_JOYHATMOTION: {
      //    1
      //
      // 8     2
      //
      //    4
      if (TRACE)
        printf("Hat %d moved to %d.\n", event.jhat.hat, event.jhat.value);

      static constexpr uint8_t JHAT_UP = 1;
      static constexpr uint8_t JHAT_DOWN = 4;
      static constexpr uint8_t JHAT_LEFT = 8;
      static constexpr uint8_t JHAT_RIGHT = 2;

      [[maybe_unused]]
      auto RisingEdge = [&](int bit) {
          return !!(event.jhat.value & bit) && !(last_jhat & bit);
        };

      // These are just mapped to the controller.
      current_gamepad &= ~(INPUT_U | INPUT_D | INPUT_L | INPUT_R);
      if (event.jhat.value & JHAT_UP) current_gamepad |= INPUT_U;
      if (event.jhat.value & JHAT_DOWN) current_gamepad |= INPUT_D;
      if (event.jhat.value & JHAT_LEFT) current_gamepad |= INPUT_L;
      if (event.jhat.value & JHAT_RIGHT) current_gamepad |= INPUT_R;

      last_jhat = event.jhat.value;
      break;
    }

    default:;
    }
  }

  return ui_dirty ? EventResult::DIRTY : EventResult::NONE;
}

void UI::Loop() {
  bool ui_dirty = true;
  for (;;) {

    if (TRACE) printf("Handle events.\n");

    switch (HandleEvents()) {
    case EventResult::EXIT: return;
    case EventResult::NONE: break;
    case EventResult::DIRTY: ui_dirty = true; break;
    }

    if (true || ui_dirty) {
      Draw();
      // printf("Flip.\n");
      SDL_Flip(screen);
      ui_dirty = false;
    }
    SDL_Delay(1);
  }
}

void UI::Draw() {
  if (TRACE) printf("Draw.\n");

  CHECK(font != nullptr);
  CHECK(drawing != nullptr);
  CHECK(screen != nullptr);

  if (view == View::PERSPECTIVE) {
    // apply perspective transform
  }

  scene.camera_pos += vel * 0.01;
  scene.Draw(drawing.get());

  drawing->BlendText32(5, 5, 0xFFFF00AA,
                       StringPrintf("Frames: %lld", frames_drawn));
  sdlutil::CopyRGBAToScreen(*drawing, screen);

  font->draw(30, 30, StringPrintf("%.5f, %.5f, %.5f",
                                  vel.x, vel.y, vel.z));

  if (mov.get()) {
    // TODO: Enqueue this.
    mov->AddFrame(*drawing);
  }

  frames_drawn++;
  if (TRACE) printf("Drew %lld.\n", frames_drawn);
}

static void InitializeSDL() {
  // Initialize SDL.
  CHECK(SDL_Init(SDL_INIT_VIDEO |
                 SDL_INIT_TIMER |
                 SDL_INIT_JOYSTICK |
                 SDL_INIT_AUDIO) >= 0);
  fprintf(stderr, "SDL initialized OK.\n");

  if (SDL_NumJoysticks() == 0) {
    printf("No joysticks were found.\n");
  } else {
    joystick = SDL_JoystickOpen(0);
    if (joystick == nullptr) {
      printf("Could not open joystick: %s\n", SDL_GetError());
    }
  }

  SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY,
                      SDL_DEFAULT_REPEAT_INTERVAL);

  SDL_EnableUNICODE(1);

  sdlutil::SetIcon("icon.png");

  screen = sdlutil::makescreen(SCREENW, SCREENH);
  CHECK(screen != nullptr);
  if (TRACE) printf("Created screen.\n");

  font = Font::Create(screen,
                      FONT_PNG,
                      FONTCHARS,
                      FONTWIDTH, FONTHEIGHT, FONTSTYLES, 1, 3);
  CHECK(font != nullptr) << "Couldn't load font.";


  font2x = Font::CreateX(2,
                         screen,
                         FONT_PNG,
                         FONTCHARS,
                         FONTWIDTH, FONTHEIGHT, FONTSTYLES, 1, 3);
  CHECK(font2x != nullptr) << "Couldn't load font.";

  font4x = Font::CreateX(4,
                         screen,
                         FONT_PNG,
                         FONTCHARS,
                         FONTWIDTH, FONTHEIGHT, FONTSTYLES, 1, 3);
  CHECK(font4x != nullptr) << "Couldn't load font.";
  if (TRACE) printf("Created fonts.\n");

  CHECK((cursor_arrow = Cursor::MakeArrow()));
  CHECK((cursor_bucket = Cursor::MakeBucket()));
  CHECK((cursor_hand = Cursor::MakeHand()));
  CHECK((cursor_hand_closed = Cursor::MakeHandClosed()));
  CHECK((cursor_eraser = Cursor::MakeEraser()));
  if (TRACE) printf("Created cursors.\n");

  SDL_SetCursor(cursor_arrow);
  SDL_ShowCursor(SDL_ENABLE);
}

int main(int argc, char **argv) {
  if (TRACE) fprintf(stderr, "In main...\n");
  ANSI::Init();

  if (TRACE) fprintf(stderr, "Try initialize SDL...\n");
  InitializeSDL();

  printf("Begin UI loop.\n");

  TriangularMesh3D mesh = LoadSTL("../platonic-dodecahedron.stl");

  UI ui(mesh);
  ui.Loop();

  printf("Quit!\n");

  SDL_Quit();
  return 0;
}

