
// Generate 2D slices of the 10D (or 8D?) parameterization.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
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
#include "color-util.h"
#include "dyson.h"
#include "image.h"
#include "lines.h"
#include "mesh.h"
#include "mov-recorder.h"
#include "mov.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "re2/re2.h"
#include "smallest-sphere.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

using namespace std;

using int64 = int64_t;
using uint32 = uint32_t;

static constexpr bool TRACE = false;

static constexpr const char *FONT_PNG = "../../cc-lib/sdl/font.png";
static Font *font = nullptr, *font2x = nullptr, *font4x = nullptr;

static SDL_Cursor *cursor_arrow = nullptr, *cursor_bucket = nullptr;
static SDL_Cursor *cursor_hand = nullptr, *cursor_hand_closed = nullptr;
static SDL_Cursor *cursor_eraser = nullptr;

#define SCREENW 1920
#define SCREENH 1080
static constexpr int IMAGE_HEIGHT = SCREENH;
static constexpr int IMAGE_WIDTH = SCREENW;
static constexpr int IMAGE_SQUARE = std::min(SCREENW, SCREENH);
static constexpr int IMAGE_TOP = SCREENH == IMAGE_SQUARE ? 0 : (IMAGE_SQUARE - IMAGE_HEIGHT) >> 1;
static constexpr int IMAGE_LEFT = SCREENW == IMAGE_SQUARE ? 0 : (IMAGE_SQUARE - IMAGE_WIDTH) >> 1  ;

static SDL_Joystick *joystick = nullptr;
static SDL_Surface *screen = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using frame3 = yocto::frame<double, 3>;
using mat3 = yocto::mat<double, 3>;
using mat4 = yocto::mat<double, 4>;

// Gamepad values, from NES!
#define INPUT_R (1<<7)
#define INPUT_L (1<<6)
#define INPUT_D (1<<5)
#define INPUT_U (1<<4)
#define INPUT_T (1<<3)
#define INPUT_S (1<<2)
#define INPUT_B (1<<1)
#define INPUT_A (1   )

static constexpr int D = 3 + 3 + 2;

// Red for negative, black for 0, green for positive.
// nominal range [-1, 1].
static constexpr ColorUtil::Gradient DISTANCE{
  /*
  GradRGB(-4.0f, 0xFFFF88),
  GradRGB(-2.0f, 0xFFFF00),
  GradRGB(-1.0f, 0xFF0000),
  GradRGB( 0.0f, 0x440044),
  GradRGB( 1.0f, 0x00FF00),
  GradRGB(+2.0f, 0x00FFFF),
  GradRGB(+4.0f, 0x88FFFF),
  */
  GradRGB(-10.0f, 0xFFFFFF),
  GradRGB(-4.0f, 0x008800),
  GradRGB( 0.0f, 0xFFFF00),
  GradRGB(+1.0f, 0x0000AA),
  GradRGB(+2.0f, 0x220044),
};


struct Scene {
  ArcFour rc = ArcFour(std::format("mri.{}", time(nullptr)));
  std::unique_ptr<Polyhedron> poly;

  std::array<double, D> current_args;
  void SetArgs(const std::array<double, D> &args) {
    current_args = args;
  }

  void Reload() {}

  static constexpr int X_AXIS = D - 2;
  static constexpr int Y_AXIS = D - 1;
  // Plot is nominally from [-XSCALE, +XSCALE].
  static constexpr double XSCALE = 0.5;
  static constexpr double YSCALE = 0.5;
  inline double Eval(const std::array<double, D> &args) {
    // PERF: A lot of this will be constant, depending on x/y axes.
    frame3 outer_frame =
      rotation_frame(vec3{1, 0, 0}, args[0]) *
      rotation_frame(vec3{0, 1, 0}, args[1]) *
      rotation_frame(vec3{0, 0, 1}, args[2]);

    frame3 inner_frame =
      rotation_frame(vec3{1, 0, 0}, args[3]) *
      rotation_frame(vec3{0, 1, 0}, args[4]) *
      rotation_frame(vec3{0, 0, 1}, args[5]);

    inner_frame.o.x = args[6];
    inner_frame.o.y = args[7];

    return LossFunction(*poly, outer_frame, inner_frame);
  }

  Scene() {
    poly.reset(new Polyhedron(Cube()));
    for (double &d : current_args) d = 0.0;
  }

  ~Scene() {
    delete poly->faces;
    poly.reset();
  }

  void Update(const vec3 &velo, const vec3 &veli) {
    // XXX mod into range
    for (int i = 0; i < 3; i++) current_args[0 + i] += velo[i];
    for (int i = 0; i < 3; i++) current_args[3 + i] += veli[i];
  }

  void Draw(ImageRGBA *img) {
    // XXX not necessary; we will set every pixel
    img->Clear32(0x000000FF);

    std::array<double, D> args = current_args;

    ParallelComp(
        IMAGE_HEIGHT,
        [&](int sy) {
          // for (int sy = 0; sy < IMAGE_HEIGHT; sy++) {
          double y = (((sy + IMAGE_TOP) / (double)IMAGE_SQUARE) * 2.0 - 1.0) * YSCALE;
          args[Y_AXIS] = y;
          for (int sx = 0; sx < IMAGE_WIDTH; sx++) {
            double x = (((sx + IMAGE_LEFT) / (double)IMAGE_SQUARE) * 2.0 - 1.0) * XSCALE;
            args[X_AXIS] = x;

            double val = Eval(args);
            img->BlendPixel32(sx, sy, ColorUtil::LinearGradient32(DISTANCE, val));
          }
        }, 12);
  }
};

struct UI {
  Scene scene;
  uint8_t current_gamepad = 0;
  int64_t frames_drawn = 0;
  uint8_t last_jhat = 0;
  // yaw, pitch, roll
  vec3 velo = vec3{0, 0, 0};
  vec3 veli = vec3{0, 0, 0};

  UI();
  void Loop();
  void Draw();

  Periodically fps_per;

  enum class EventResult { NONE, DIRTY, EXIT, };
  EventResult HandleEvents();

  std::unique_ptr<ImageRGBA> drawing;
  int mousex = 0, mousey = 0;
  bool dragging = false;

  std::unique_ptr<MovRecorder> mov;
};

UI::UI() : fps_per(1.0 / 60.0) {
  drawing.reset(new ImageRGBA(SCREENW, SCREENH));
  CHECK(drawing != nullptr);
  drawing->Clear32(0x000000FF);
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
      case 1:
        // left and right
        velo.x = MapAxis(j->value);
        break;
      case 0:
        // forward and back
        velo.y = MapAxis(j->value);
        break;

      case 4:
        veli.x = MapAxis(j->value);
        break;

      case 3:
        veli.y = MapAxis(j->value);
        break;

        // TODO: Support roll
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

      case SDLK_r: {
        scene.Reload();
        ui_dirty = true;
      }

      case SDLK_HOME: {
        // TODO: Reset pos

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

      case SDLK_m: {
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

    scene.Update(velo * 0.01, veli * 0.01);

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

  scene.Draw(drawing.get());

  drawing->BlendText32(5, 5, 0xFFFF00AA,
                       StringPrintf("Frames: %lld", frames_drawn));
  sdlutil::CopyRGBAToScreen(*drawing, screen);

  font->draw(30, 30, StringPrintf("%.5f, %.5f, %.5f",
                                  velo.x, velo.y, velo.z));
  font->draw(30, 50, StringPrintf("%.5f, %.5f, %.5f",
                                  veli.x, veli.y, veli.z));

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

  UI ui;
  ui.Loop();

  printf("Quit!\n");

  SDL_Quit();
  return 0;
}

