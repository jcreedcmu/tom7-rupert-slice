
default: viewer.exe shrinkplacer.exe

all: viewer.exe

CC_LIB=../../cc-lib
SQLITE=../../sqlite
CODEC=../../codec

# ASAN=-fsanitize=address
# OPT=-g -O2
OPT=-g -O3
FLTO=-flto=thin

# SDLOPATH=SDL/build
# SDLOPATH=c:/code/SDL/build
SDLSDK=c:/code/SDL
SDLOPATH=$(SDLSDK)/build
# SDLINCLUDES=-ISDL/include
SDLINCLUDES=-I$(SDLSDK)/include

# -fno-strict-aliasing
# -I/usr/local/include -I/usr/include
CXXFLAGS=$(ASAN) $(OPT) -m64 -D__MINGW32__ $(SDLINCLUDES) $(FLTO) -Wall -Wno-unused-function -I. -I.. -I$(CC_LIB) -I$(SQLITE) -I$(CODEC) --std=c++23

RUPERT_OBJECTS=../polyhedra.o ../mesh.o ../solutions.o ../hull3d.o ../hull.o ../dyson.o ../smallest-sphere.o

SQLITE_OBJECTS=$(SQLITE)/sqlite3.o $(SQLITE)/database.o

SDLUTILOBJECTS=$(CC_LIB)/sdl/sdlutil.o $(CC_LIB)/sdl/font.o $(CC_LIB)/sdl/cursor.o
SDLOBJECTS=$(SDLOPATH)/SDL.o $(SDLOPATH)/SDL_error.o $(SDLOPATH)/SDL_fatal.o $(SDLOPATH)/SDL_audio.o $(SDLOPATH)/SDL_audiocvt.o $(SDLOPATH)/SDL_audiodev.o $(SDLOPATH)/SDL_mixer.o $(SDLOPATH)/SDL_mixer_MMX.o $(SDLOPATH)/SDL_mixer_MMX_VC.o $(SDLOPATH)/SDL_mixer_m68k.o $(SDLOPATH)/SDL_wave.o $(SDLOPATH)/SDL_cdrom.o $(SDLOPATH)/SDL_cpuinfo.o $(SDLOPATH)/SDL_active.o $(SDLOPATH)/SDL_events.o $(SDLOPATH)/SDL_expose.o $(SDLOPATH)/SDL_keyboard.o $(SDLOPATH)/SDL_mouse.o $(SDLOPATH)/SDL_quit.o $(SDLOPATH)/SDL_resize.o $(SDLOPATH)/SDL_rwops.o $(SDLOPATH)/SDL_getenv.o $(SDLOPATH)/SDL_iconv.o $(SDLOPATH)/SDL_malloc.o $(SDLOPATH)/SDL_qsort.o $(SDLOPATH)/SDL_stdlib.o $(SDLOPATH)/SDL_string.o $(SDLOPATH)/SDL_thread.o $(SDLOPATH)/SDL_timer.o $(SDLOPATH)/SDL_RLEaccel.o $(SDLOPATH)/SDL_blit.o $(SDLOPATH)/SDL_blit_0.o $(SDLOPATH)/SDL_blit_1.o $(SDLOPATH)/SDL_blit_A.o $(SDLOPATH)/SDL_blit_N.o $(SDLOPATH)/SDL_bmp.o $(SDLOPATH)/SDL_cursor.o $(SDLOPATH)/SDL_gamma.o $(SDLOPATH)/SDL_pixels.o $(SDLOPATH)/SDL_stretch.o $(SDLOPATH)/SDL_surface.o $(SDLOPATH)/SDL_video.o $(SDLOPATH)/SDL_yuv.o $(SDLOPATH)/SDL_yuv_mmx.o $(SDLOPATH)/SDL_yuv_sw.o $(SDLOPATH)/SDL_joystick.o $(SDLOPATH)/SDL_nullevents.o $(SDLOPATH)/SDL_nullmouse.o $(SDLOPATH)/SDL_nullvideo.o $(SDLOPATH)/SDL_diskaudio.o $(SDLOPATH)/SDL_dummyaudio.o $(SDLOPATH)/SDL_sysevents.o $(SDLOPATH)/SDL_sysmouse.o $(SDLOPATH)/SDL_syswm.o $(SDLOPATH)/SDL_wingl.o $(SDLOPATH)/SDL_dibevents.o $(SDLOPATH)/SDL_dibvideo.o $(SDLOPATH)/SDL_dx5events.o $(SDLOPATH)/SDL_dx5video.o $(SDLOPATH)/SDL_dx5yuv.o $(SDLOPATH)/SDL_dibaudio.o $(SDLOPATH)/SDL_dx5audio.o $(SDLOPATH)/SDL_mmjoystick.o $(SDLOPATH)/SDL_syscdrom.o $(SDLOPATH)/SDL_sysmutex.o $(SDLOPATH)/SDL_syssem.o $(SDLOPATH)/SDL_systhread.o $(SDLOPATH)/SDL_syscond.o $(SDLOPATH)/SDL_systimer.o $(SDLOPATH)/SDL_sysloadso.o $(SDLUTILOBJECTS) sdl-win32-main.o

LINKSDL=  -lm -mthreads -luser32 -lgdi32 -lwinmm -ldxguid

RE2_OBJECTS=$(CC_LIB)/re2/bitstate.o $(CC_LIB)/re2/compile.o $(CC_LIB)/re2/dfa.o $(CC_LIB)/re2/filtered_re2.o $(CC_LIB)/re2/mimics_pcre.o $(CC_LIB)/re2/nfa.o $(CC_LIB)/re2/onepass.o $(CC_LIB)/re2/parse.o $(CC_LIB)/re2/perl_groups.o $(CC_LIB)/re2/prefilter.o $(CC_LIB)/re2/prefilter_tree.o $(CC_LIB)/re2/prog.o $(CC_LIB)/re2/re2.o $(CC_LIB)/re2/regexp.o $(CC_LIB)/re2/set.o $(CC_LIB)/re2/simplify.o $(CC_LIB)/re2/stringpiece.o $(CC_LIB)/re2/tostring.o $(CC_LIB)/re2/unicode_casefold.o $(CC_LIB)/re2/unicode_groups.o $(CC_LIB)/re2/util/rune.o $(CC_LIB)/re2/util/strutil.o

UTIL_OBJECTS=$(CC_LIB)/util.o $(CC_LIB)/arcfour.o $(CC_LIB)/base/stringprintf.o $(CC_LIB)/base/logging.o $(CC_LIB)/stb_image.o $(CC_LIB)/stb_image_write.o $(CC_LIB)/stb_truetype.o $(CC_LIB)/color-util.o $(CC_LIB)/image.o $(CC_LIB)/crypt/sha256.o $(CC_LIB)/ansi.o $(CC_LIB)/ansi-image.o $(CC_LIB)/image-resize.o $(CC_LIB)/zip.o $(CC_LIB)/miniz.o $(CC_LIB)/png.o

CODEC_OBJECTS=$(CODEC)/mov.o $(CODEC)/mov-recorder.o

%.o : %.cc *.h makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@
	echo -n "."

$(CC_LIB)/%.o : $(CC_LIB)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

../%.o : ../%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

$(SQLITE)/%.o : $(SQLITE)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

OBJECTS=$(UTIL_OBJECTS) $(RUPERT_OBJECTS) $(RE2_OBJECTS)

# XXX can maybe use miniz instead of zlib here? Is it for SDL?
LFLAGS= $(ASAN) -L. $(LINKSDL) -m64 -Wl,--subsystem,console -lz $(OPT) $(FLTO) -static

viewer.exe : viewer.o $(OBJECTS) $(SDLOBJECTS) $(SQLITE_OBJECTS) $(CODEC_OBJECTS)
	$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

shrinkplacer.exe : shrinkplacer.o $(OBJECTS) $(SDLOBJECTS) $(SQLITE_OBJECTS) $(CODEC_OBJECTS)
	$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

mri.exe : mri.o $(OBJECTS) $(SDLOBJECTS) $(SQLITE_OBJECTS) $(CODEC_OBJECTS)
	$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

clean :
	rm -f *.o $(OBJECTS) *.exe

veryclean :
	rm -f *.o $(OBJECTS) $(SDLOBJECTS) *.exe
