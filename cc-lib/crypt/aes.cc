/*

This code is based on "Tiny AES in C" by kokke (github), who placed
it in the public domain.

---

This is an implementation of the AES algorithm, specifically ECB, CTR
and CBC mode. Block size can be chosen in aes.h - available choices
are AES128, AES192, AES256.

The implementation is verified against the test vectors in:
  National Institute of Standards and Technology Special Publication
  800-38A 2001 ED

ECB-AES128
----------

  plain-text:
    6bc1bee22e409f96e93d7e117393172a
    ae2d8a571e03ac9c9eb76fac45af8e51
    30c81c46a35ce411e5fbc1191a0a52ef
    f69f2445df4f9b17ad2b417be66c3710

  key:
    2b7e151628aed2a6abf7158809cf4f3c

  resulting cipher
    3ad77bb40d7a3660a89ecaf32466ef97
    f5d3d58503b9699de785895a96fdbaaf
    43b1cd7f598ece23881b00e3ed030688
    7b0c785e27e8ad3f8223207104725dd4


NOTE: String length must be evenly divisible by 16 bytes.
      You should pad the end of the string with zeros if this
      is not the case. For AES192/256 the key size is proportionally
      larger.
*/

#include "aes.h"
#include <cstdint>
#include <string.h>

using uint8 = uint8_t;
using uint32 = uint32_t;

// The number of columns comprising a state in AES. This is a constant
// in AES.
static constexpr int Nb = 4;

// state - array holding the intermediate results during decryption.
typedef uint8 state_t[4][4];

// The lookup-tables are marked const so they can be placed in
// read-only storage instead of RAM. The numbers below can be computed
// dynamically trading ROM for RAM - This can be useful in (embedded)
// bootloader applications, where ROM is often limited.
static constexpr uint8 sbox[256] = {
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
  0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
  0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
  0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
  0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
  0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
  0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
  0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
  0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
  0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
  0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
  0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
  0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
  0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
  0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
  0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
  0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

static constexpr uint8 rsbox[256] = {
  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
  0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
  0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
  0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
  0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
  0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
  0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
  0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
  0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
  0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
  0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
  0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
  0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
  0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
  0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
  0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
  0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };

// The round constant word array, Rcon[i], contains the values given by
// x to the power (i-1) being powers of x (x is denoted as {02}) in
// the field GF(2^8)
static constexpr uint8 Rcon[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

/*
 * Jordan Goulder points out in PR #12
 * (https://github.com/kokke/tiny-AES-C/pull/12), that you can remove
 * most of the elements in the Rcon array, because they are unused.
 *
 * From Wikipedia's article on the Rijndael key schedule @
 * https://en.wikipedia.org/wiki/Rijndael_key_schedule#Rcon
 *
 * "Only the first some of these constants are actually used – up to
 *  rcon[10] for AES-128 (as 11 round keys are needed), up to rcon[8]
 *  for AES-192, up to rcon[7] for AES-256. rcon[0] is not used in AES
 *  algorithm."
 */


#define getSBoxValue(num) (sbox[(num)])
#define getSBoxInvert(num) (rsbox[(num)])

// This function produces Nb(NUM_ROUNDS+1) round keys. The round keys are used
// in each round to decrypt the states.
template<int KEY_WORDS, int NUM_ROUNDS>
static void KeyExpansion(uint8 *round_key, const uint8 *Key) {
  uint8 tempa[4]; // Used for the column/row operations

  // The first round key is the key itself.
  for (int i = 0; i < KEY_WORDS; ++i) {
    round_key[(i * 4) + 0] = Key[(i * 4) + 0];
    round_key[(i * 4) + 1] = Key[(i * 4) + 1];
    round_key[(i * 4) + 2] = Key[(i * 4) + 2];
    round_key[(i * 4) + 3] = Key[(i * 4) + 3];
  }

  // All other round keys are found from the previous round keys.
  for (int i = KEY_WORDS; i < Nb * (NUM_ROUNDS + 1); ++i) {
    {
      const int k = (i - 1) * 4;
      tempa[0] = round_key[k + 0];
      tempa[1] = round_key[k + 1];
      tempa[2] = round_key[k + 2];
      tempa[3] = round_key[k + 3];
    }

    if (i % KEY_WORDS == 0) {
      // This function shifts the 4 bytes in a word to the left once.
      // [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

      // Function RotWord()
      {
        const uint8 u8tmp = tempa[0];
        tempa[0] = tempa[1];
        tempa[1] = tempa[2];
        tempa[2] = tempa[3];
        tempa[3] = u8tmp;
      }

      // SubWord() is a function that takes a four-byte input word and
      // applies the S-box to each of the four bytes to produce an output word.

      // Function Subword()
      {
        tempa[0] = getSBoxValue(tempa[0]);
        tempa[1] = getSBoxValue(tempa[1]);
        tempa[2] = getSBoxValue(tempa[2]);
        tempa[3] = getSBoxValue(tempa[3]);
      }

      tempa[0] ^= Rcon[i / KEY_WORDS];
    }

    // Is this AES256?
    if (KEY_WORDS == 8) {
      if (i % KEY_WORDS == 4) {
        // Function Subword()
        tempa[0] = getSBoxValue(tempa[0]);
        tempa[1] = getSBoxValue(tempa[1]);
        tempa[2] = getSBoxValue(tempa[2]);
        tempa[3] = getSBoxValue(tempa[3]);
      }
    }

    const int j = i * 4;
    const int k = (i - KEY_WORDS) * 4;
    round_key[j + 0] = round_key[k + 0] ^ tempa[0];
    round_key[j + 1] = round_key[k + 1] ^ tempa[1];
    round_key[j + 2] = round_key[k + 2] ^ tempa[2];
    round_key[j + 3] = round_key[k + 3] ^ tempa[3];
  }
}

template<int KEYBITS>
void AES<KEYBITS>::InitCtx(struct Ctx *ctx, const uint8 *key) {
  KeyExpansion<KEY_WORDS, NUM_ROUNDS>(ctx->round_key, key);
}

template<int KEYBITS>
void AES<KEYBITS>::InitCtxIV(struct Ctx *ctx,
                             const uint8 *key, const uint8 *iv) {
  KeyExpansion<KEY_WORDS, NUM_ROUNDS>(ctx->round_key, key);
  memcpy(ctx->iv, iv, BLOCKLEN);
}

template<int KEYBITS>
void AES<KEYBITS>::Ctx_set_iv(struct Ctx *ctx, const uint8 *iv) {
  memcpy(ctx->iv, iv, BLOCKLEN);
}

// This function adds the round key to state.
// The round key is added to the state by an XOR function.
static void AddRoundKey(uint8 round, state_t *state,
                        const uint8 *round_key) {
  for (uint8 i = 0; i < 4; ++i) {
    for (uint8 j = 0; j < 4; ++j) {
      (*state)[i][j] ^= round_key[(round * Nb * 4) + (i * Nb) + j];
    }
  }
}

// The SubBytes Function Substitutes the values in the
// state matrix with values in an S-box.
static void SubBytes(state_t *state) {
  for (uint8 i = 0; i < 4; ++i) {
    for (uint8 j = 0; j < 4; ++j) {
      (*state)[j][i] = getSBoxValue((*state)[j][i]);
    }
  }
}

// The ShiftRows() function shifts the rows in the state to the left.
// Each row is shifted with different offset.
// Offset = Row number. So the first row is not shifted.
static void ShiftRows(state_t *state) {
  // Rotate first row 1 columns to left
  uint8 temp     = (*state)[0][1];
  (*state)[0][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[3][1];
  (*state)[3][1] = temp;

  // Rotate second row 2 columns to left
  temp           = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;

  temp           = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;

  // Rotate third row 3 columns to left
  temp           = (*state)[0][3];
  (*state)[0][3] = (*state)[3][3];
  (*state)[3][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[1][3];
  (*state)[1][3] = temp;
}

static constexpr uint8 xtime(uint8 x) {
  return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}

// MixColumns function mixes the columns of the state matrix
static void MixColumns(state_t *state) {
  for (uint8 i = 0; i < 4; ++i) {
    uint8 t = (*state)[i][0];
    uint8 Tmp =
      (*state)[i][0] ^ (*state)[i][1] ^ (*state)[i][2] ^ (*state)[i][3];
    uint8 Tm =
      (*state)[i][0] ^ (*state)[i][1];
    Tm = xtime(Tm);
    (*state)[i][0] ^= Tm ^ Tmp;
    Tm = (*state)[i][1] ^ (*state)[i][2];
    Tm = xtime(Tm);
    (*state)[i][1] ^= Tm ^ Tmp;
    Tm = (*state)[i][2] ^ (*state)[i][3];
    Tm = xtime(Tm);
    (*state)[i][2] ^= Tm ^ Tmp;
    Tm = (*state)[i][3] ^ t;
    Tm = xtime(Tm);
    (*state)[i][3] ^= Tm ^ Tmp;
  }
}

// Mul is used to multiply numbers in the field GF(2^8).
// Note: The last call to xtime() is unneeded, but often ends up
// generating a smaller binary. The compiler seems to be able to
// vectorize the operation better this way.
static constexpr uint8 Mul(uint8 x, uint8 y) {
  return (((y & 1) * x) ^
          ((y >> 1 & 1) * xtime(x)) ^
          ((y >> 2 & 1) * xtime(xtime(x))) ^
          ((y >> 3 & 1) * xtime(xtime(xtime(x)))) ^
          ((y >> 4 & 1) * xtime(xtime(xtime(xtime(x))))));
}

// MixColumns function mixes the columns of the state matrix. The
// method used to multiply may be difficult to understand for the
// inexperienced. Please use the references to gain more information.
static void InvMixColumns(state_t *state) {
  for (int i = 0; i < 4; ++i) {
    const uint8 a = (*state)[i][0];
    const uint8 b = (*state)[i][1];
    const uint8 c = (*state)[i][2];
    const uint8 d = (*state)[i][3];

    (*state)[i][0] = Mul(a, 0x0e) ^ Mul(b, 0x0b) ^ Mul(c, 0x0d) ^ Mul(d, 0x09);
    (*state)[i][1] = Mul(a, 0x09) ^ Mul(b, 0x0e) ^ Mul(c, 0x0b) ^ Mul(d, 0x0d);
    (*state)[i][2] = Mul(a, 0x0d) ^ Mul(b, 0x09) ^ Mul(c, 0x0e) ^ Mul(d, 0x0b);
    (*state)[i][3] = Mul(a, 0x0b) ^ Mul(b, 0x0d) ^ Mul(c, 0x09) ^ Mul(d, 0x0e);
  }
}


// The SubBytes Function Substitutes the values in the
// state matrix with values in an S-box.
static void InvSubBytes(state_t *state) {
  for (uint8 i = 0; i < 4; ++i) {
    for (uint8 j = 0; j < 4; ++j) {
      (*state)[j][i] = getSBoxInvert((*state)[j][i]);
    }
  }
}

static void InvShiftRows(state_t *state) {
  // Rotate first row 1 columns to right
  uint8 temp = (*state)[3][1];
  (*state)[3][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[0][1];
  (*state)[0][1] = temp;

  // Rotate second row 2 columns to right
  temp = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;

  temp = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;

  // Rotate third row 3 columns to right
  temp = (*state)[0][3];
  (*state)[0][3] = (*state)[1][3];
  (*state)[1][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[3][3];
  (*state)[3][3] = temp;
}


// Cipher is the main function that encrypts the PlainText.
template<int NUM_ROUNDS>
static void Cipher(state_t *state, const uint8 *round_key) {
  // Add the First round key to the state before starting the rounds.
  AddRoundKey(0, state, round_key);

  // There will be NUM_ROUNDS rounds.
  // The first NUM_ROUNDS-1 rounds are identical.
  // These NUM_ROUNDS-1 rounds are executed in the loop below.
  for (uint8 round = 1; round < NUM_ROUNDS; ++round) {
    SubBytes(state);
    ShiftRows(state);
    MixColumns(state);
    AddRoundKey(round, state, round_key);
  }

  // The last round is given below.
  // The MixColumns function is not here in the last round.
  SubBytes(state);
  ShiftRows(state);
  AddRoundKey(NUM_ROUNDS, state, round_key);
}

template<int NUM_ROUNDS>
static void InvCipher(state_t *state, const uint8 *round_key) {
  // Add the First round key to the state before starting the rounds.
  AddRoundKey(NUM_ROUNDS, state, round_key);

  // There will be NUM_ROUNDS rounds.
  // The first NUM_ROUNDS-1 rounds are identical.
  // These NUM_ROUNDS-1 rounds are executed in the loop below.
  for (uint8 round = NUM_ROUNDS - 1; round > 0; --round) {
    InvShiftRows(state);
    InvSubBytes(state);
    AddRoundKey(round, state, round_key);
    InvMixColumns(state);
  }

  // The last round is given below.
  // The MixColumns function is not here in the last round.
  InvShiftRows(state);
  InvSubBytes(state);
  AddRoundKey(0, state, round_key);
}


/*****************************************************************************/
/* Public functions:                                                         */
/*****************************************************************************/

template<int KEYBITS>
void AES<KEYBITS>::EncryptECB(const struct Ctx *ctx, uint8 *buf) {
  // The next function call encrypts the PlainText with the Key using
  // AES algorithm.
  Cipher<NUM_ROUNDS>((state_t*)buf, ctx->round_key);
}

template<int KEYBITS>
void AES<KEYBITS>::DecryptECB(const struct Ctx *ctx, uint8 *buf) {
  // The next function call decrypts the PlainText with the Key using
  // AES algorithm.
  InvCipher<NUM_ROUNDS>((state_t*)buf, ctx->round_key);
}

template<int BYTES>
static void XorWithIv(uint8 *buf, const uint8 *iv) {
  for (uint8 i = 0; i < BYTES; ++i) {
    // The block in AES is always 128bit no matter the key size
    buf[i] ^= iv[i];
  }
}

template<int KEYBITS>
void AES<KEYBITS>::EncryptCBC(struct Ctx *ctx, uint8 *buf, uint32 length) {
  uint8 *iv = ctx->iv;
  for (uintptr_t i = 0; i < length; i += BLOCKLEN) {
    XorWithIv<BLOCKLEN>(buf, iv);
    Cipher<NUM_ROUNDS>((state_t*)buf, ctx->round_key);
    iv = buf;
    buf += BLOCKLEN;
  }
  /* store iv in ctx for next call */
  memcpy(ctx->iv, iv, BLOCKLEN);
}

template<int KEYBITS>
void AES<KEYBITS>::DecryptCBC(struct Ctx *ctx, uint8 *buf, uint32 length) {
  uint8 storeNextIv[BLOCKLEN];
  for (uintptr_t i = 0; i < length; i += BLOCKLEN) {
    memcpy(storeNextIv, buf, BLOCKLEN);
    InvCipher<NUM_ROUNDS>((state_t*)buf, ctx->round_key);
    XorWithIv<BLOCKLEN>(buf, ctx->iv);
    memcpy(ctx->iv, storeNextIv, BLOCKLEN);
    buf += BLOCKLEN;
  }
}


/* Symmetrical operation: same function for encrypting as for
   decrypting. Note any IV/nonce should never be reused with the same
   key */
template<int KEYBITS>
void AES<KEYBITS>::XcryptCTR(struct Ctx *ctx, uint8 *buf, uint32 length) {
  uint8 buffer[BLOCKLEN];

  int bi = BLOCKLEN;
  for (uint32 i = 0; i < length; ++i, ++bi) {
    if (bi == BLOCKLEN) {
      /* we need to regen xor compliment in buffer */
      memcpy(buffer, ctx->iv, BLOCKLEN);
      Cipher<NUM_ROUNDS>((state_t*)buffer, ctx->round_key);

      /* Increment iv and handle overflow */
      for (bi = BLOCKLEN - 1; bi >= 0; --bi) {
        /* inc will overflow */
        if (ctx->iv[bi] == 255) {
          ctx->iv[bi] = 0;
          continue;
        }
        ctx->iv[bi]++;
        break;
      }
      bi = 0;
    }

    buf[i] ^= buffer[bi];
  }
}

template struct AES<128>;
template struct AES<192>;
template struct AES<256>;
