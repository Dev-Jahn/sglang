/* SPDX-License-Identifier: Apache-2.0 */
#ifndef QWEN4_PLE_DISK_FETCHER_H
#define QWEN4_PLE_DISK_FETCHER_H

#include <stddef.h>
#include <stdint.h>

#define PLE_FETCHER_ABI_VERSION 1U
#define PLE_FETCHER_WAIT_NS 100000000L
#define PLE_FETCHER_TOTAL_READ_WAITS 50U
#define PLE_FETCHER_QUIESCE_WAITS 10U
#define PLE_FETCHER_READ_WAITS (PLE_FETCHER_TOTAL_READ_WAITS - PLE_FETCHER_QUIESCE_WAITS)
#define PLE_FETCHER_LOCK_BUDGET_MS 5500U
#define PLE_FETCHER_FAILURE_NONE 0U
#define PLE_FETCHER_FAILURE_SETUP 1U
#define PLE_FETCHER_FAILURE_REGISTER_BUFFER 2U
#define PLE_FETCHER_FAILURE_REGISTER_FILE 3U
#define PLE_FETCHER_DESTROY_BUSY_WAITS 5000U
/* The destroy drain waits at most 50 times for 100 ms, about five seconds. */
#define PLE_FETCHER_DESTROY_DRAIN_WAITS 50U

static inline int ple_fetcher_retry_after_timeout(unsigned* timeouts) {
  ++*timeouts;
  return *timeouts < PLE_FETCHER_READ_WAITS;
}

/* Calls on a handle must be externally serialized. A single caller owns the
 * handle, and no read or last_error call may be in flight during destroy. */
unsigned ple_fetcher_abi_version(void);
unsigned ple_fetcher_lock_budget_ms(void);
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
void ple_fetcher_test_interrupt_submissions(unsigned interrupts);
void ple_fetcher_test_deadline_ms(unsigned milliseconds);
void ple_fetcher_test_expire_completion_deadline(unsigned count);
int ple_fetcher_test_ring_open(void* opaque);
#endif

#endif
