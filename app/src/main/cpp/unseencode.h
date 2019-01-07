#ifndef _UNSEENCODE_H_INCLUDED_
#define _UNSEENCODE_H_INCLUDED_

#include <stdint.h>

extern const char *CHARSET;
#define ENCODE_LEN 6

// BCH(511, 121)
#define MSG_BIT 114
#define CSUM_BIT 7
#define ECC_BIT 390
#define TOTAL_BIT (MSG_BIT + CSUM_BIT + ECC_BIT)
#define CRC_POLY 0xd5

uint8_t *unseencode_encode(const char* msg);
char *unseencode_decode(const uint8_t *code);

#endif
