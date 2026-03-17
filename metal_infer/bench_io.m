/*
 * bench_io.m — Benchmark pread into Metal buffers (cold vs warm)
 * Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation -lpthread bench_io.m -o bench_io
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

#define EXPERT_SIZE 7077888

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

typedef struct { int fd; void *dst; off_t offset; } PTask;

static void *pread_fn(void *arg) {
    PTask *t = (PTask *)arg;
    pread(t->fd, t->dst, EXPERT_SIZE, t->offset);
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const char *path = "/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/packed_experts/layer_00.bin";

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    int K = 4;
    int offsets[4] = {0, 100, 200, 300};
    int rounds = 10;

    id<MTLBuffer> metal_bufs[4];
    for (int k = 0; k < K; k++) {
        metal_bufs[k] = [device newBufferWithLength:EXPERT_SIZE
                            options:MTLResourceStorageModeShared];
    }

    printf("=== Sequential pread (4 experts, same offsets each round) ===\n");
    for (int r = 0; r < rounds; r++) {
        double t0 = now_ms();
        for (int k = 0; k < K; k++) {
            pread(fd, [metal_bufs[k] contents], EXPERT_SIZE, (off_t)offsets[k] * EXPERT_SIZE);
        }
        double t1 = now_ms();
        printf("  Round %2d: %5.2f ms (%5.1f GB/s)\n", r, t1-t0,
               (K * EXPERT_SIZE / 1e9) / ((t1-t0) / 1000.0));
    }

    printf("\n=== Parallel pread (4 threads, same offsets) ===\n");
    for (int r = 0; r < rounds; r++) {
        PTask tasks[4];
        pthread_t threads[4];
        double t0 = now_ms();
        for (int k = 0; k < K; k++) {
            tasks[k].fd = fd;
            tasks[k].dst = [metal_bufs[k] contents];
            tasks[k].offset = (off_t)offsets[k] * EXPERT_SIZE;
            pthread_create(&threads[k], NULL, pread_fn, &tasks[k]);
        }
        for (int k = 0; k < K; k++) pthread_join(threads[k], NULL);
        double t1 = now_ms();
        printf("  Round %2d: %5.2f ms (%5.1f GB/s)\n", r, t1-t0,
               (K * EXPERT_SIZE / 1e9) / ((t1-t0) / 1000.0));
    }

    printf("\n=== Parallel pread (4 threads, RANDOM offsets per round) ===\n");
    srand(42);
    for (int r = 0; r < rounds; r++) {
        int rnd[4] = { rand()%512, rand()%512, rand()%512, rand()%512 };
        PTask tasks[4]; pthread_t threads[4];
        double t0 = now_ms();
        for (int k = 0; k < K; k++) {
            tasks[k].fd = fd; tasks[k].dst = [metal_bufs[k] contents];
            tasks[k].offset = (off_t)rnd[k] * EXPERT_SIZE;
            pthread_create(&threads[k], NULL, pread_fn, &tasks[k]);
        }
        for (int k = 0; k < K; k++) pthread_join(threads[k], NULL);
        double t1 = now_ms();
        printf("  Round %2d: %5.2f ms (exp %d,%d,%d,%d) (%5.1f GB/s)\n",
               r, t1-t0, rnd[0], rnd[1], rnd[2], rnd[3],
               (K * EXPERT_SIZE / 1e9) / ((t1-t0) / 1000.0));
    }

    printf("\n=== GCD dispatch_apply (4 experts, warm) ===\n");
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    for (int r = 0; r < rounds; r++) {
        PTask tasks[4];
        for (int k = 0; k < K; k++) {
            tasks[k].fd = fd; tasks[k].dst = [metal_bufs[k] contents];
            tasks[k].offset = (off_t)offsets[k] * EXPERT_SIZE;
        }
        PTask *tp = tasks;
        double t0 = now_ms();
        dispatch_apply(K, q, ^(size_t i) {
            pread(tp[i].fd, tp[i].dst, EXPERT_SIZE, tp[i].offset);
        });
        double t1 = now_ms();
        printf("  Round %2d: %5.2f ms (%5.1f GB/s)\n", r, t1-t0,
               (K * EXPERT_SIZE / 1e9) / ((t1-t0) / 1000.0));
    }

    // Simulate 60 layers × 4 experts with random offsets across ALL layer files
    printf("\n=== 60-layer simulation (4 random experts per layer, sequential files) ===\n");
    int layer_fds[60];
    for (int l = 0; l < 60; l++) {
        char fpath[512];
        snprintf(fpath, sizeof(fpath),
                 "/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/packed_experts/layer_%02d.bin", l);
        layer_fds[l] = open(fpath, O_RDONLY);
    }
    srand(123);
    for (int r = 0; r < 3; r++) {
        double t_total = 0;
        for (int l = 0; l < 60; l++) {
            int rnd[4] = { rand()%512, rand()%512, rand()%512, rand()%512 };
            PTask tasks[4]; pthread_t threads[4];
            double t0 = now_ms();
            for (int k = 0; k < K; k++) {
                tasks[k].fd = layer_fds[l]; tasks[k].dst = [metal_bufs[k] contents];
                tasks[k].offset = (off_t)rnd[k] * EXPERT_SIZE;
                pthread_create(&threads[k], NULL, pread_fn, &tasks[k]);
            }
            for (int k = 0; k < K; k++) pthread_join(threads[k], NULL);
            t_total += now_ms() - t0;
        }
        printf("  Token %d: %.1f ms total (%.2f ms/layer, %.1f tok/s I/O-limited)\n",
               r, t_total, t_total/60, 1000.0/t_total);
    }
    for (int l = 0; l < 60; l++) close(layer_fds[l]);

    close(fd);
    return 0;
}
