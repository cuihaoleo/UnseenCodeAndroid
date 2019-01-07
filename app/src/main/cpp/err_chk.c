#include "err_chk.h"
#include "bch.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 *  Primitive polynomial copied from MATLAB
 *  (R2018a/toolbox/comm/comm/primpoly.m)
 *  init_bch requires 5 <= m <=16 
 */
static const unsigned int BCH_PRIM_POLY[] = \
    { 0, 0, 0, 0, 0, 37, 67, 137, 285, 529,
      1033, 2053, 4179, 8219, 17475, 32771 };

inline static size_t highestBitIndexU8(uint8_t n) {
    size_t r = 0;
    while (n >>= 1) r++;
    return r;
}

inline static size_t highestBitIndexU64(uint64_t n) {
    size_t r = 0;
    while (n >>= 1) r++;
    return r;
}

static uint8_t *packBufferBCH(const uint8_t *stream, int msg_bit,
                              struct bch_control *bch) {
    int msg_byte = DIV_ROUND_UP(msg_bit, 8);
    int ecc_bits = bch->ecc_bits;
    int ecc_byte = bch->ecc_bytes;
    uint8_t *buf = (uint8_t*)calloc(msg_byte + ecc_byte + 1, sizeof(uint8_t));

    uint8_t *msg_buf = buf;
    uint8_t *ecc_buf = buf + msg_byte;
    int msg_pad = (msg_byte << 3) - msg_bit;

    for (int i=0, j=msg_pad; i<msg_bit; i++, j++)
        msg_buf[j >> 3] |= stream[i] << (7 - (j & 7));

    for (int i=0; i<ecc_bits; i++)
        ecc_buf[i >> 3] |= stream[i + msg_bit] << (7 - (i & 7));

    return buf;
}

static uint8_t *unpackBufferBCH(const uint8_t *buf, int msg_bit, struct bch_control *bch) {
    uint8_t *stream = (uint8_t*)calloc(msg_bit + bch->ecc_bits + 1, sizeof(uint8_t));
    int ecc_bits = bch->ecc_bits;
    int msg_byte = DIV_ROUND_UP(msg_bit, 8);

    const uint8_t *msg_buf = buf;
    const uint8_t *ecc_buf = buf + msg_byte;
    int msg_pad = (msg_byte << 3) - msg_bit;
    int i, j;

    for (i=0, j=msg_pad; i<msg_bit; i++, j++)
        stream[i] = 1 & (msg_buf[j >> 3] >> (7 - (j & 7)));

    for (j=0; i<msg_bit+ecc_bits; i++, j++)
        stream[i] = 1 & (ecc_buf[j >> 3] >> (7 - (j & 7)));

    return stream;
}

uint8_t *encodeBCH(const uint8_t *stream, int data_bit, int ecc_bit) {
    int n = data_bit + ecc_bit;                     // n, msg length
    unsigned int m = highestBitIndexU64(n + 1);     // GF(m)
    assert(m >= 3 && m < ARRAY_SIZE(BCH_PRIM_POLY));
    assert((n & (n + 1)) == 0);
    int k = ecc_bit, t;
    struct bch_control *bch = NULL;

    for (t = 1 << (m-3); t > 0; t--) {
        bch = init_bch(m, t, BCH_PRIM_POLY[m]);

        if (bch != NULL && (int)bch->ecc_bits == k)
            break;

        free_bch(bch);
        bch = NULL;
    }

    if (bch == NULL)
        return NULL;

    uint8_t *exp_stream = (uint8_t*)calloc(data_bit + ecc_bit, sizeof(uint8_t));
    for (int i=0; i<data_bit; i++)
        exp_stream[i] = stream[i] & 1;

    int data_byte = DIV_ROUND_UP(data_bit, 8);
    uint8_t *buf = packBufferBCH(exp_stream, n-k, bch);

    encode_bch(bch, buf, data_byte, buf + data_byte);
    uint8_t *ret = unpackBufferBCH(buf, data_bit, bch);

    free_bch(bch);
    free(buf);
    free(exp_stream);

    return ret;
}

uint8_t *decodeBCH(const uint8_t *stream,
                   int data_bit, int ecc_bit) {
    int n = data_bit + ecc_bit;                     // n, msg length
    unsigned int m = highestBitIndexU64(n + 1);     // GF(m)
    assert(m >= 3 && m < sizeof(BCH_PRIM_POLY) / sizeof(BCH_PRIM_POLY[0]));
    assert((n & (n + 1)) == 0);
    int k = ecc_bit, t;
    struct bch_control *bch = NULL;

    for (t = 1 << (m-3); t > 0; t--) {
        bch = init_bch(m, t, BCH_PRIM_POLY[m]);

        if (bch != NULL && (int)bch->ecc_bits == k)
            break;

        free_bch(bch);
        bch = NULL;
    }

    if (bch == NULL)
        return NULL;

    int data_byte = DIV_ROUND_UP(data_bit, 8), n_err;
    unsigned int *errloc = (unsigned int*)calloc(t, sizeof(unsigned int));

    uint8_t *buf = packBufferBCH(stream, n-k, bch);
    n_err = decode_bch(bch, buf, data_byte, buf + data_byte, NULL, NULL, errloc);

    for (int i=0; i<n_err; i++) {
        int loc = errloc[i];
        if (loc < data_bit)
            buf[loc >> 3] ^= 1 << (loc & 7);
    }

    uint8_t *ret = unpackBufferBCH(buf, data_bit, bch);

    free(errloc);
    free_bch(bch);
    free(buf);

    return ret;
}

int checkCRC(uint8_t *buf, size_t data_len, uint64_t poly, int check_only) {
    size_t n = highestBitIndexU64(poly);
    uint8_t *ret = (uint8_t*)malloc(data_len + n);
    uint8_t *p;
    size_t i;

    memcpy(ret, buf, data_len + n);
    for (i = 0; i < data_len; i++) {
        if (ret[i]) {
            p = ret + i + n;
            uint64_t t = poly;
            do {
                *(p--) ^= t & 1;
            } while (t >>= 1);
        }
    }

    if (!check_only)
        memcpy(buf + data_len, ret + data_len, n);

    p = ret + data_len;
    for (i=0; i<n; i++)
        if (p[i])
            break;

    free(ret);
    return i >= n;
}
