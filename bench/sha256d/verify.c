/* SPDX-License-Identifier: GPL-2.0-only */
/* Link this offline check to the exact SHA objects used in one cpuminer build. */
#include "algo/sha/sha256d.h"
#include <openssl/sha.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(SHA256D_FORCE_SHA_NI) && defined(SHA256D_SHA)
#define EXPECTED_SCANNER scanhash_sha256d_sha
#define SCANNER_NAME "SHA-NI"
enum { SCAN_WIDTH = 2 };
#elif !defined(SHA256D_FORCE_SHA_NI) && defined(SHA256D_16WAY)
#define EXPECTED_SCANNER scanhash_sha256d_16way
#define SCANNER_NAME "AVX-512"
enum { SCAN_WIDTH = 16 };
#else
#error "Expected the selected SHA256d scanner to match the benchmark build"
#endif

enum { HEADERS = 32 };
static const uint32_t genesis[20] = {
    0x01000000,0,0,0,0,0,0,0,0,0x3ba3edfd,0x7a7b12b2,0x7ac72c3e,
    0x67768f61,0x7fc81bc3,0x888a5132,0x3a9fb8aa,0x4b1e5e4a,0x29ab5f49,
    0xffff001d,0x1dac2b7c
};

/* The scanner calls these normal miner services; record solutions without a pool. */
bool opt_benchmark = false;
static struct work_restart restart_flag;
struct work_restart *work_restart = &restart_flag;
static uint32_t first_nonce;
static uint32_t expected[HEADERS][8];
static bool submitted[HEADERS];

static int compare256(const uint32_t *a, const uint32_t *b)
{
    for (int i = 7; i >= 0; --i)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

bool valid_hash(const void *hash, const void *target)
{
    return compare256(hash, target) <= 0;
}

bool submit_solution(struct work *work, const void *hash, struct thr_info *thread)
{
    const uint32_t nonce = work->data[19];
    const uint32_t index = nonce - first_nonce;
    if (thread->id != 0 || index >= HEADERS || submitted[index] ||
        memcmp(hash, expected[index], 32) != 0) {
        fprintf(stderr, "SHA256d verification failed for nonce word %08" PRIx32 "\n", nonce);
        exit(EXIT_FAILURE);
    }
    submitted[index] = true;
    return true;
}

static void prepare_reference(struct work *work)
{
    uint8_t header[80], digest[32];
    for (int i = 0; i < HEADERS; ++i) {
        for (int j = 0; j < 20; ++j) {
            /* cpuminer stores big-endian SHA input words in native integer order. */
            uint32_t word = j == 19 ? first_nonce + (uint32_t)i : work->data[j];
            word = __builtin_bswap32(word);
            memcpy(header + 4*j, &word, 4);
        }
        if (!SHA256(header, sizeof header, digest) ||
            !SHA256(digest, sizeof digest, (unsigned char *)expected[i]))
            exit(EXIT_FAILURE);
    }
}

static void check_target(const algo_gate_t *gate, struct work *work, const uint32_t *target)
{
    struct thr_info thread = {0};
    uint64_t hashes_done = 0;
    memset(submitted, 0, sizeof submitted);
    memcpy(work->target, target, 32);
    work->data[19] = first_nonce;
    /* Each scanner subtracts its batch width from max_nonce before comparing. */
    gate->scanhash(work, first_nonce + HEADERS + SCAN_WIDTH, &hashes_done, &thread);
    if (hashes_done != HEADERS || work->data[19] != first_nonce + HEADERS) {
        fprintf(stderr, "Unexpected SHA256d scan count: %" PRIu64 "\n", hashes_done);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < HEADERS; ++i) {
        if (submitted[i] != (compare256(expected[i], target) <= 0)) {
            fprintf(stderr, "Wrong target decision for nonce word %08" PRIx32 "\n", first_nonce + i);
            exit(EXIT_FAILURE);
        }
    }
}

int main(void)
{
    algo_gate_t gate = {0};
    struct work work = {0};
    if (!register_sha256d_algo(&gate) || gate.scanhash != EXPECTED_SCANNER) {
        fputs("SHA256d " SCANNER_NAME " scanner was not selected\n", stderr);
        return EXIT_FAILURE;
    }
    for (int lane = 0; lane < 16; ++lane) {
        uint32_t target[8];
        memcpy(work.data, genesis, sizeof genesis);
        first_nonce = genesis[19] - (uint32_t)lane;
        prepare_reference(&work);

        /* The easy target forces all eight digest words through the rare path. */
        memset(target, 0xff, sizeof target);
        check_target(&gate, &work, target);
        /* Genesis spans all SIMD positions and its high word equals zero. */
        memcpy(target, expected[lane], sizeof target);
        check_target(&gate, &work, target);
        /* Genesis digest's least significant word is nonzero. */
        --target[0];
        check_target(&gate, &work, target);
        /* Vary the common 64-byte prefix, tail and threshold as well. */
        work.data[0] ^= 1u << lane;
        work.data[17] += (uint32_t)lane + 1;
        first_nonce = (uint32_t)lane * 0x01010101u;
        prepare_reference(&work);
        check_target(&gate, &work, expected[(lane + 7) % HEADERS]);
    }
    puts("SHA256d " SCANNER_NAME " verification OK (64 scans, 2048 header checks)");
    return EXIT_SUCCESS;
}
