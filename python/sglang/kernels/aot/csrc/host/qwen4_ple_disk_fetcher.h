#ifndef QWEN4_PLE_DISK_FETCHER_H
#define QWEN4_PLE_DISK_FETCHER_H

#include <stddef.h>
#include <stdint.h>

#define PLE_FETCHER_WAIT_NS 100000000L
#define PLE_FETCHER_MAX_WAITS 50U
#define PLE_FETCHER_DESTROY_MAX_WAITS 5000U

static inline int ple_fetcher_retry_after_timeout(unsigned* timeouts) {
  ++*timeouts;
  return *timeouts < PLE_FETCHER_MAX_WAITS;
}

void* ple_fetcher_create(
    int file_fd, void* buffer, size_t buffer_bytes, unsigned max_pages, int register_buffer, int* failure_stage);
int ple_fetcher_read(void* opaque, const uint64_t* offsets, unsigned count, void* buffer, size_t buffer_bytes);
int ple_fetcher_last_error(void* opaque, unsigned* index, int* result);
int ple_fetcher_destroy(void* opaque);

#ifdef PLE_FETCHER_TESTING
void ple_fetcher_test_limit_submissions(unsigned pages);
unsigned ple_fetcher_test_partial_submissions(void);
void ple_fetcher_test_stall_completions(int enabled);
void ple_fetcher_test_stall_wakes(unsigned wakes);
void ple_fetcher_test_successful_empty_wakes(unsigned wakes);
void ple_fetcher_test_completion_on_last_wake(int enabled);
#endif

#endif
