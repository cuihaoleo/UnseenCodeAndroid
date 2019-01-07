#ifndef __BLUEMARK_ERR_CHK_H__
#define __BLUEMARK_ERR_CHK_H__

#include <stdint.h>
#include <stddef.h>

uint8_t *encodeBCH(const uint8_t *stream, int data_bit, int ecc_bit);

uint8_t *decodeBCH(const uint8_t *stream, int data_bit, int ecc_bit);

int checkCRC(uint8_t *buf, size_t data_len, uint64_t poly, int check_only);

#endif
