#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static const uint32_t rounds[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
static const uint32_t initial[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static uint32_t rotate_right(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}

static void compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t words[64];
    for (unsigned i = 0; i < 16; ++i) {
        const uint8_t *bytes = block + 4 * i;
        words[i] = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16)
                 | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
    }
    for (unsigned i = 16; i < 64; ++i) {
        uint32_t left = words[i - 15], right = words[i - 2];
        uint32_t small0 = rotate_right(left, 7) ^ rotate_right(left, 18) ^ (left >> 3);
        uint32_t small1 = rotate_right(right, 17) ^ rotate_right(right, 19) ^ (right >> 10);
        words[i] = words[i - 16] + small0 + words[i - 7] + small1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (unsigned i = 0; i < 64; ++i) {
        uint32_t large1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t first = h + large1 + choose + rounds[i] + words[i];
        uint32_t large0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = d + first;
        d = c; c = b; b = a; a = first + large0 + majority;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void digest_message(const uint8_t *data, size_t length, uint8_t output[32]) {
    uint32_t state[8];
    memcpy(state, initial, sizeof state);
    uint64_t bit_length = (uint64_t)length * 8;
    while (length >= 64) {
        compress(state, data);
        data += 64;
        length -= 64;
    }
    uint8_t final[128] = {0};
    if (length) memcpy(final, data, length);
    final[length] = 0x80;
    size_t final_length = length < 56 ? 64 : 128;
    for (unsigned i = 0; i < 8; ++i) {
        final[final_length - 1 - i] = (uint8_t)(bit_length >> (8 * i));
    }
    compress(state, final);
    if (final_length == 128) compress(state, final + 64);
    for (unsigned i = 0; i < 8; ++i) {
        output[4 * i] = (uint8_t)(state[i] >> 24);
        output[4 * i + 1] = (uint8_t)(state[i] >> 16);
        output[4 * i + 2] = (uint8_t)(state[i] >> 8);
        output[4 * i + 3] = (uint8_t)state[i];
    }
}

/* Four independent messages share each SIMD compression instruction. */
typedef uint32_t word4 __attribute__((vector_size(16)));

static word4 rotate4(word4 value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}

static uint32_t read_word(const uint8_t *bytes) {
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16)
         | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static void compress4(word4 state[8], const uint8_t *blocks[4], word4 active) {
    word4 words[64];
    for (unsigned i = 0; i < 16; ++i) {
        words[i] = (word4){read_word(blocks[0] + 4 * i), read_word(blocks[1] + 4 * i),
                          read_word(blocks[2] + 4 * i), read_word(blocks[3] + 4 * i)};
    }
    for (unsigned i = 16; i < 64; ++i) {
        word4 left = words[i - 15], right = words[i - 2];
        word4 small0 = rotate4(left, 7) ^ rotate4(left, 18) ^ (left >> 3);
        word4 small1 = rotate4(right, 17) ^ rotate4(right, 19) ^ (right >> 10);
        words[i] = words[i - 16] + small0 + words[i - 7] + small1;
    }
    word4 a = state[0], b = state[1], c = state[2], d = state[3];
    word4 e = state[4], f = state[5], g = state[6], h = state[7];
    for (unsigned i = 0; i < 64; ++i) {
        word4 large1 = rotate4(e, 6) ^ rotate4(e, 11) ^ rotate4(e, 25);
        word4 choose = (e & f) ^ (~e & g);
        word4 first = h + large1 + choose + rounds[i] + words[i];
        word4 large0 = rotate4(a, 2) ^ rotate4(a, 13) ^ rotate4(a, 22);
        word4 majority = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = d + first;
        d = c; c = b; b = a; a = first + large0 + majority;
    }
    word4 result[8] = {a, b, c, d, e, f, g, h};
    for (unsigned i = 0; i < 8; ++i) {
        state[i] = (active & (state[i] + result[i])) | (~active & state[i]);
    }
}

static void digest_four(const uint8_t *data[4], const uint64_t lengths[4], uint8_t *output) {
    word4 state[8];
    for (unsigned i = 0; i < 8; ++i) {
        state[i] = (word4){initial[i], initial[i], initial[i], initial[i]};
    }
    uint8_t tails[4][128] = {{0}}, zero[64] = {0};
    uint64_t full_blocks[4], total_blocks[4], maximum_blocks = 0;
    for (unsigned lane = 0; lane < 4; ++lane) {
        full_blocks[lane] = lengths[lane] / 64;
        unsigned remainder = (unsigned)(lengths[lane] % 64);
        if (remainder) memcpy(tails[lane], data[lane] + full_blocks[lane] * 64, remainder);
        tails[lane][remainder] = 0x80;
        unsigned tail_length = remainder < 56 ? 64 : 128;
        uint64_t bits = lengths[lane] * 8;
        for (unsigned i = 0; i < 8; ++i) {
            tails[lane][tail_length - 1 - i] = (uint8_t)(bits >> (8 * i));
        }
        total_blocks[lane] = full_blocks[lane] + tail_length / 64;
        if (total_blocks[lane] > maximum_blocks) maximum_blocks = total_blocks[lane];
    }
    for (uint64_t block = 0; block < maximum_blocks; ++block) {
        const uint8_t *blocks[4];
        word4 active;
        for (unsigned lane = 0; lane < 4; ++lane) {
            active[lane] = block < total_blocks[lane] ? UINT32_MAX : 0;
            blocks[lane] = block < full_blocks[lane] ? data[lane] + 64 * block
                         : block < total_blocks[lane] ? tails[lane] + 64 * (block - full_blocks[lane])
                         : zero;
        }
        compress4(state, blocks, active);
    }
    for (unsigned lane = 0; lane < 4; ++lane) {
        for (unsigned word = 0; word < 8; ++word) {
            uint32_t value = state[word][lane];
            output[lane * 32 + word * 4] = (uint8_t)(value >> 24);
            output[lane * 32 + word * 4 + 1] = (uint8_t)(value >> 16);
            output[lane * 32 + word * 4 + 2] = (uint8_t)(value >> 8);
            output[lane * 32 + word * 4 + 3] = (uint8_t)value;
        }
    }
}

static double elapsed(struct timespec before, struct timespec after) {
    return (double)(after.tv_sec - before.tv_sec)
         + (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
}

/* ctypes.CDLL releases the interpreter lock for this entire packet call. */
int irg_sha256_packet(const uint8_t *payload, uint64_t payload_size,
                      const uint64_t *offsets, const uint64_t *lengths,
                      uint64_t count, uint8_t *output,
                      double *wall_seconds, double *thread_cpu_seconds) {
    struct timespec wall_before, wall_after, cpu_before, cpu_after;
    if (!wall_seconds || !thread_cpu_seconds ||
        (count && (!offsets || !lengths || !output)) ||
        (payload_size && !payload)) return 1;
    if (clock_gettime(CLOCK_MONOTONIC, &wall_before) ||
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_before)) return 2;
    uint64_t i = 0;
    for (; count - i >= 4; i += 4) {
        const uint8_t *messages[4];
        for (unsigned lane = 0; lane < 4; ++lane) {
            uint64_t position = i + lane;
            if (offsets[position] > payload_size || lengths[position] > payload_size - offsets[position]) return 1;
            messages[lane] = payload ? payload + offsets[position] : NULL;
        }
        digest_four(messages, lengths + i, output + 32 * i);
    }
    for (; i < count; ++i) {
        if (offsets[i] > payload_size || lengths[i] > payload_size - offsets[i]) return 1;
        digest_message(payload ? payload + offsets[i] : NULL, (size_t)lengths[i], output + 32 * i);
    }
    if (clock_gettime(CLOCK_MONOTONIC, &wall_after) ||
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_after)) return 2;
    *wall_seconds = elapsed(wall_before, wall_after);
    *thread_cpu_seconds = elapsed(cpu_before, cpu_after);
    return 0;
}
