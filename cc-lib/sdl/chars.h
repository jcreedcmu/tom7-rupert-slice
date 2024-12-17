
#ifndef _CC_LIB_SDLUTIL_CHARS_H
#define _CC_LIB_SDLUTIL_CHARS_H

// This describes the font in font.png, which is used in many of my
// projects.

// TODO: Should probably do some work to clean this up or expose it a
// different way, since it's easy for these defines to conflict with
// other symbols (especially the color ones). The tricky thing is that
// we really want these as macros, since we use them with string
// concatenation at lexing time.

#define WHITE "^0"
#define BLUE "^1"
#define RED "^2"
#define YELLOW "^3"
#define GREY "^4"
#define GRAY GREY
#define GREEN "^5"
#define POP "^<"

#define ALPHA100 "^#"
#define ALPHA50 "^$"
#define ALPHA25 "^%"

/* there are some non-ascii symbols in the font */
#define CHECKMARK "\xF2"
#define ESC "\xF3"
#define HEART "\xF4"
/* here L means "long" */
#define LCMARK1 "\xF5"
#define LCMARK2 "\xF6"
#define LCHECKMARK LCMARK1 LCMARK2
#define LRARROW1 "\xF7"
#define LRARROW2 "\xF8"
#define LRARROW LRARROW1 LRARROW2
#define LLARROW1 "\xF9"
#define LLARROW2 "\xFA"
#define LLARROW LLARROW1 LLARROW2

/* BAR_0 ... BAR_10 are guaranteed to be consecutive */
#define BAR_0 "\xE0"
#define BAR_1 "\xE1"
#define BAR_2 "\xE2"
#define BAR_3 "\xE3"
#define BAR_4 "\xE4"
#define BAR_5 "\xE5"
#define BAR_6 "\xE6"
#define BAR_7 "\xE7"
#define BAR_8 "\xE8"
#define BAR_9 "\xE9"
#define BAR_10 "\xEA"
#define BARSTART "\xEB"

#define FONTCHARS " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`-=[]\\;',./~!@#$%^&*()_+{}|:\"<>?" CHECKMARK ESC HEART LCMARK1 LCMARK2 BAR_0 BAR_1 BAR_2 BAR_3 BAR_4 BAR_5 BAR_6 BAR_7 BAR_8 BAR_9 BAR_10 BARSTART LRARROW LLARROW

/* additionally, one style just holds little helper
   images instead of letters */
#define PICS "^9"

#define FONTSTYLES 8

#define FONTWIDTH 9
#define FONTHEIGHT 16
#define FONTOVERLAP 1

#endif
