#include "unseencode.h"
#include "err_chk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

const char *CHARSET = " _abcdefghijklmnopqrstuvwxyz" \
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

void obfuscate(uint8_t *buf, size_t len) {
    uint32_t x = len;
    for (size_t i=0; i!=len; i++) {
        x = x * 1103515245 + 12345;
        buf[i] ^= x & 1;
    }
}

uint8_t *unseencode_encode(const char* msg) {
    int msg_len = strnlen(msg, MSG_BIT / 8);
    uint8_t *csum_buf = calloc(MSG_BIT + CSUM_BIT, sizeof(uint8_t));
    uint8_t *ret = NULL;

    assert(strlen(CHARSET) <= (1 << ENCODE_LEN));
    assert(MSG_BIT % ENCODE_LEN == 0);

    for (int i=0; i<msg_len; i++) {
        char *pos;
        if ((pos = strchr(CHARSET, msg[i])) == NULL) {
            fprintf(stderr, "Bad character in message!");
            goto done;
        }

        for (int j=0, enc=pos-CHARSET; j<8 && enc; j++, enc>>=1)
            csum_buf[i*8 + j] = enc & 1;
    }

    checkCRC(csum_buf, MSG_BIT, CRC_POLY, 0);
    ret = encodeBCH(csum_buf, MSG_BIT + CSUM_BIT, ECC_BIT);
    obfuscate(ret, TOTAL_BIT);

done:
    free(csum_buf);
    return ret;
}


char *unseencode_decode(const uint8_t *code) {
    uint8_t buf[TOTAL_BIT];
    uint8_t *csum_buf = NULL;
    char *ret = NULL, *p;

    memcpy(buf, code, sizeof(buf));
    obfuscate(buf, TOTAL_BIT);
    csum_buf = decodeBCH(buf, MSG_BIT + CSUM_BIT, ECC_BIT);

    if (csum_buf == NULL)
        goto done;

    if (!checkCRC(csum_buf, MSG_BIT, CRC_POLY, 1))
        goto done;

    ret = calloc(MSG_BIT / 8 + 1, sizeof(char));
    p = ret;
    for (int i=0; i+7<MSG_BIT; i+=8) {
        uint8_t num = 0;

        for (int j=7; j>=0; j--)
            num = (num << 1) + csum_buf[i+j];

        *(p++) = CHARSET[num];
    }

done:
    if (csum_buf != NULL)
        free(csum_buf);

    return ret;
}
