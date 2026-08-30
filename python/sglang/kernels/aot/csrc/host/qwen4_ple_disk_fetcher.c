#define _GNU_SOURCE
#include "qwen4_ple_disk_fetcher.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include "qwen4_ple_io_uring_abi.h"

struct fetcher {
  int ring_fd;
  int fixed_buffer;
  int poisoned;
  int state;
  int has_last_error;
  unsigned last_error_index;
  int last_error_result;
  unsigned max_pages;
  void* registered_buffer;
  size_t registered_buffer_bytes;
  unsigned *sq_head, *sq_tail, *sq_mask, *sq_entries, *sq_array;
  struct io_uring_sqe* sqes;
  unsigned *cq_head, *cq_tail, *cq_mask;
  struct io_uring_cqe* cqes;
  void *sq_ptr, *cq_ptr, *sqes_ptr;
  size_t sq_size, cq_size, sqes_size;
  int single_mmap;
};

#ifndef EUCLEAN
#define EUCLEAN 117
#endif

enum fetcher_failure_stage {
  FETCHER_FAILURE_NONE = 0,
  FETCHER_FAILURE_SETUP = 1,
  FETCHER_FAILURE_REGISTER_BUFFER = 2,
  FETCHER_FAILURE_REGISTER_FILE = 3,
};

static long setup(unsigned entries, struct io_uring_params* p) {
  return syscall(__NR_io_uring_setup, entries, p);
}
static long enter_ring(int fd, unsigned submit, unsigned wait, unsigned flags) {
  return syscall(__NR_io_uring_enter, fd, submit, wait, flags, NULL, 0);
}
static long submit_ring(struct fetcher* f, unsigned submit) {
#ifdef PLE_FETCHER_TESTING
  extern unsigned ple_test_submit_limit;
  extern unsigned ple_test_partial_count;
  if (ple_test_submit_limit && submit > ple_test_submit_limit) {
    submit = ple_test_submit_limit;
    ++ple_test_partial_count;
  }
#endif
  return enter_ring(f->ring_fd, submit, 0, 0);
}
static long enter_ring_bounded(struct fetcher* f) {
#ifdef PLE_FETCHER_TESTING
  extern int ple_test_stall_completion;
  extern unsigned ple_test_stall_wakes;
  extern unsigned ple_test_successful_empty_wakes;
  extern int ple_test_completion_on_last_wake;
  if (ple_test_stall_completion) {
    errno = ETIME;
    return -1;
  }
  if (ple_test_stall_wakes) {
    --ple_test_stall_wakes;
    errno = ETIME;
    return -1;
  }
  if (ple_test_successful_empty_wakes) {
    --ple_test_successful_empty_wakes;
    if (ple_test_successful_empty_wakes || !ple_test_completion_on_last_wake) return 0;
    ple_test_completion_on_last_wake = 0;
  }
#endif
  struct qwen4_ple_kernel_timespec timeout = {
      .tv_sec = 0,
      .tv_nsec = PLE_FETCHER_WAIT_NS,
  };
  struct io_uring_getevents_arg arg = {
      .ts = (uintptr_t)&timeout,
  };
  return syscall(
      __NR_io_uring_enter, f->ring_fd, 0, 1, IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG, &arg, sizeof(arg));
}
static long register_ring(int fd, unsigned op, const void* arg, unsigned nr) {
  return syscall(__NR_io_uring_register, fd, op, arg, nr);
}

void* ple_fetcher_create(
    int file_fd, void* buffer, size_t buffer_bytes, unsigned max_pages, int register_buffer, int* failure_stage) {
  if (failure_stage) *failure_stage = FETCHER_FAILURE_NONE;
  if (!buffer || !max_pages || buffer_bytes < (size_t)max_pages * 4096 || ((uintptr_t)buffer & 4095)) {
    errno = EINVAL;
    return NULL;
  }
  struct fetcher* f = calloc(1, sizeof(*f));
  if (!f) return NULL;
  f->ring_fd = -1;
  f->max_pages = max_pages;
  f->fixed_buffer = !!register_buffer;
  f->registered_buffer = buffer;
  f->registered_buffer_bytes = buffer_bytes;
  struct io_uring_params p = {0};
  _Static_assert(sizeof(struct io_uring_sqe) == 64, "invalid io_uring SQE ABI");
  _Static_assert(sizeof(struct io_uring_cqe) == 16, "invalid io_uring CQE ABI");
  _Static_assert(sizeof(struct io_sqring_offsets) == 40, "invalid io_uring SQ offsets ABI");
  _Static_assert(sizeof(struct io_cqring_offsets) == 40, "invalid io_uring CQ offsets ABI");
  _Static_assert(sizeof(struct io_uring_params) == 120, "invalid io_uring params ABI");
  _Static_assert(sizeof(struct io_uring_getevents_arg) == 24, "invalid io_uring getevents ABI");
  if (failure_stage) *failure_stage = FETCHER_FAILURE_SETUP;
  f->ring_fd = setup(max_pages, &p);
  if (f->ring_fd < 0) goto fail;
  if (!(p.features & IORING_FEAT_EXT_ARG)) {
    errno = EOPNOTSUPP;
    goto fail;
  }
  f->sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  f->cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
  f->single_mmap = !!(p.features & IORING_FEAT_SINGLE_MMAP);
  if (f->single_mmap && f->cq_size > f->sq_size) f->sq_size = f->cq_size;
  f->sq_ptr = mmap(NULL, f->sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, f->ring_fd, IORING_OFF_SQ_RING);
  if (f->sq_ptr == MAP_FAILED) goto fail;
  if (f->single_mmap) {
    f->cq_ptr = f->sq_ptr;
  } else {
    f->cq_ptr =
        mmap(NULL, f->cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, f->ring_fd, IORING_OFF_CQ_RING);
    if (f->cq_ptr == MAP_FAILED) goto fail;
  }
  f->sqes_size = p.sq_entries * sizeof(struct io_uring_sqe);
  f->sqes_ptr =
      mmap(NULL, f->sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, f->ring_fd, IORING_OFF_SQES);
  if (f->sqes_ptr == MAP_FAILED) goto fail;
  f->sqes = f->sqes_ptr;
  f->sq_head = f->sq_ptr + p.sq_off.head;
  f->sq_tail = f->sq_ptr + p.sq_off.tail;
  f->sq_mask = f->sq_ptr + p.sq_off.ring_mask;
  f->sq_entries = f->sq_ptr + p.sq_off.ring_entries;
  f->sq_array = f->sq_ptr + p.sq_off.array;
  f->cq_head = f->cq_ptr + p.cq_off.head;
  f->cq_tail = f->cq_ptr + p.cq_off.tail;
  f->cq_mask = f->cq_ptr + p.cq_off.ring_mask;
  f->cqes = f->cq_ptr + p.cq_off.cqes;
  struct iovec iov = {buffer, buffer_bytes};
  if (f->fixed_buffer) {
    if (failure_stage) *failure_stage = FETCHER_FAILURE_REGISTER_BUFFER;
    if (register_ring(f->ring_fd, IORING_REGISTER_BUFFERS, &iov, 1) < 0) goto fail;
  }
  if (failure_stage) *failure_stage = FETCHER_FAILURE_REGISTER_FILE;
  if (register_ring(f->ring_fd, IORING_REGISTER_FILES, &file_fd, 1) < 0) goto fail;
  if (failure_stage) *failure_stage = FETCHER_FAILURE_NONE;
  return f;
fail: {
  int saved = errno;
  if (f->sqes_ptr && f->sqes_ptr != MAP_FAILED) munmap(f->sqes_ptr, f->sqes_size);
  if (!f->single_mmap && f->cq_ptr && f->cq_ptr != MAP_FAILED) munmap(f->cq_ptr, f->cq_size);
  if (f->sq_ptr && f->sq_ptr != MAP_FAILED) munmap(f->sq_ptr, f->sq_size);
  if (f->ring_fd >= 0) close(f->ring_fd);
  free(f);
  errno = saved;
  return NULL;
}
}

static unsigned reap_available(struct fetcher* f, unsigned limit, int* result) {
#ifdef PLE_FETCHER_TESTING
  extern int ple_test_stall_completion;
  extern unsigned ple_test_stall_wakes;
  extern unsigned ple_test_successful_empty_wakes;
  if (ple_test_stall_completion) return 0;
  if (ple_test_stall_wakes) return 0;
  if (ple_test_successful_empty_wakes) return 0;
#endif
  unsigned completed = 0;
  unsigned head = __atomic_load_n(f->cq_head, __ATOMIC_RELAXED);
  unsigned tail = __atomic_load_n(f->cq_tail, __ATOMIC_ACQUIRE);
  while (head != tail && completed < limit) {
    struct io_uring_cqe* cqe = &f->cqes[head & *f->cq_mask];
    if (cqe->res != 4096 && *result == 0) {
      *result = cqe->res < 0 ? cqe->res : -EIO;
      f->has_last_error = 1;
      f->last_error_index = (unsigned)cqe->user_data;
      f->last_error_result = cqe->res;
    }
    ++head;
    ++completed;
  }
  if (completed) __atomic_store_n(f->cq_head, head, __ATOMIC_RELEASE);
  return completed;
}

static int reap_bounded(struct fetcher* f, unsigned count, int* result, unsigned* waits) {
  unsigned completed = 0;
  while (completed < count) {
    unsigned reaped = reap_available(f, count - completed, result);
    completed += reaped;
    if (completed == count) return 0;
    if (reaped) continue;
    long rc;
    do {
      rc = enter_ring_bounded(f);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0 && errno != ETIME) return -errno;
    reaped = reap_available(f, count - completed, result);
    completed += reaped;
    if (completed == count) return 0;
    if (reaped) continue;
    if (!ple_fetcher_retry_after_timeout(waits)) return -ETIMEDOUT;
  }
  return 0;
}

static int poison_fetcher(struct fetcher* f) {
  __atomic_store_n(&f->poisoned, 1, __ATOMIC_RELEASE);
  if (f->ring_fd >= 0) {
    int ring_fd = f->ring_fd;
    f->ring_fd = -1;
    munmap(f->sqes_ptr, f->sqes_size);
    f->sqes_ptr = NULL;
    if (!f->single_mmap) {
      munmap(f->cq_ptr, f->cq_size);
    }
    f->cq_ptr = NULL;
    munmap(f->sq_ptr, f->sq_size);
    f->sq_ptr = NULL;
    // The kernel keeps page references for in-flight requests after close starts
    // cancellation. Their buffers remain valid until each request completes.
    close(ring_fd);
  }
  return -EUCLEAN;
}

static int quiesce_after_error(struct fetcher* f, unsigned submitted, unsigned count, unsigned completed) {
  unsigned attempts = 0;
  while (submitted < count && attempts++ < PLE_FETCHER_MAX_WAITS) {
    long rc;
    do {
      rc = submit_ring(f, count - submitted);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) break;
    submitted += (unsigned)rc;
  }

  int ignored_result = 0;
  unsigned quiesce_waits = 0;
  if (submitted > completed && reap_bounded(f, submitted - completed, &ignored_result, &quiesce_waits) < 0) {
    return poison_fetcher(f);
  }
  if (submitted != count) {
    return poison_fetcher(f);
  }
  return 0;
}

static int finish_read(struct fetcher* f, int result) {
  __atomic_store_n(&f->state, 0, __ATOMIC_RELEASE);
  return result;
}

int ple_fetcher_read(void* opaque, const uint64_t* offsets, unsigned count, void* buffer, size_t buffer_bytes) {
  struct fetcher* f = opaque;
  if (!f || !buffer || count > f->max_pages || ((uintptr_t)buffer & 4095)) return -EINVAL;
  if (count == 0) return 0;
  if (!offsets) return -EINVAL;
  size_t required_bytes = (size_t)count * 4096;
  if (buffer_bytes < required_bytes) return -EFAULT;
  if (f->fixed_buffer && (buffer != f->registered_buffer || required_bytes > f->registered_buffer_bytes))
    return -EFAULT;
  if (__atomic_load_n(&f->poisoned, __ATOMIC_ACQUIRE)) return -EUCLEAN;
  int expected_state = 0;
  if (!__atomic_compare_exchange_n(&f->state, &expected_state, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return expected_state == 2 ? -EBADF : -EBUSY;
  f->has_last_error = 0;
  f->last_error_index = 0;
  f->last_error_result = 0;
  unsigned tail = __atomic_load_n(f->sq_tail, __ATOMIC_RELAXED);
  unsigned head = __atomic_load_n(f->sq_head, __ATOMIC_ACQUIRE);
  if (tail - head + count > *f->sq_entries) return finish_read(f, -ENOSPC);
  for (unsigned index = 0; index < count; ++index) {
    unsigned position = tail & *f->sq_mask;
    struct io_uring_sqe* sqe = &f->sqes[position];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = f->fixed_buffer ? IORING_OP_READ_FIXED : IORING_OP_READ;
    sqe->fd = 0;
    sqe->off = offsets[index];
    sqe->addr = (uintptr_t)buffer + (size_t)index * 4096;
    sqe->len = 4096;
    if (f->fixed_buffer) sqe->buf_index = 0;
    sqe->flags = IOSQE_FIXED_FILE;
    sqe->user_data = index;
    f->sq_array[position] = position;
    ++tail;
  }
  __atomic_store_n(f->sq_tail, tail, __ATOMIC_RELEASE);
  long rc;
  unsigned submitted = 0;
  unsigned waits = 0;
  while (submitted < count) {
    do {
      rc = submit_ring(f, count - submitted);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0 || rc == 0) {
      int error = rc < 0 ? -errno : -EIO;
      if (quiesce_after_error(f, submitted, count, 0) < 0) return finish_read(f, -EUCLEAN);
      return finish_read(f, error);
    }
    submitted += (unsigned)rc;
  }
  unsigned completed = 0;
  int result = 0;
  while (completed < count) {
    unsigned reaped = reap_available(f, count - completed, &result);
    completed += reaped;
    if (completed == count) break;
    if (!reaped) {
      do {
        rc = enter_ring_bounded(f);
      } while (rc < 0 && errno == EINTR);
      if (rc < 0 && errno != ETIME) {
        int error = -errno;
        if (quiesce_after_error(f, count, count, completed) < 0) return finish_read(f, -EUCLEAN);
        return finish_read(f, error);
      }
      reaped = reap_available(f, count - completed, &result);
      completed += reaped;
      if (completed == count) break;
      if (reaped) continue;
      if (!ple_fetcher_retry_after_timeout(&waits)) {
        if (quiesce_after_error(f, count, count, completed) < 0) return finish_read(f, -EUCLEAN);
        return finish_read(f, -ETIMEDOUT);
      }
      continue;
    }
  }
  return finish_read(f, result);
}

int ple_fetcher_last_error(void* opaque, unsigned* index, int* result) {
  struct fetcher* f = opaque;
  if (!f || !index || !result) return -EINVAL;
  int expected_state = 0;
  if (!__atomic_compare_exchange_n(&f->state, &expected_state, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return expected_state == 2 ? -EBADF : -EBUSY;
  int has_error = f->has_last_error;
  if (has_error) {
    *index = f->last_error_index;
    *result = f->last_error_result;
  }
  finish_read(f, 0);
  return has_error;
}

int ple_fetcher_destroy(void* opaque) {
  struct fetcher* f = opaque;
  if (!f) return 0;
  int expected_state = 0;
  unsigned waits = 0;
  while (!__atomic_compare_exchange_n(&f->state, &expected_state, 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    if (expected_state == 2) return 0;
    if (++waits >= PLE_FETCHER_DESTROY_MAX_WAITS) return -EBUSY;
    expected_state = 0;
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000L};
    nanosleep(&pause, NULL);
  }
  if (f->ring_fd >= 0) {
    register_ring(f->ring_fd, IORING_UNREGISTER_FILES, NULL, 0);
    if (f->fixed_buffer) register_ring(f->ring_fd, IORING_UNREGISTER_BUFFERS, NULL, 0);
  }
  if (f->sqes_ptr) munmap(f->sqes_ptr, f->sqes_size);
  if (!f->single_mmap && f->cq_ptr) munmap(f->cq_ptr, f->cq_size);
  if (f->sq_ptr) munmap(f->sq_ptr, f->sq_size);
  if (f->ring_fd >= 0) close(f->ring_fd);
  free(f);
  return 0;
}

#ifdef PLE_FETCHER_TESTING
unsigned ple_test_submit_limit = 0;
unsigned ple_test_partial_count = 0;
int ple_test_stall_completion = 0;
unsigned ple_test_stall_wakes = 0;
unsigned ple_test_successful_empty_wakes = 0;
int ple_test_completion_on_last_wake = 0;

void ple_fetcher_test_limit_submissions(unsigned pages) {
  ple_test_submit_limit = pages;
  ple_test_partial_count = 0;
}

unsigned ple_fetcher_test_partial_submissions(void) {
  return ple_test_partial_count;
}

void ple_fetcher_test_stall_completions(int enabled) {
  ple_test_stall_completion = !!enabled;
}

void ple_fetcher_test_stall_wakes(unsigned wakes) {
  ple_test_stall_wakes = wakes;
}

void ple_fetcher_test_successful_empty_wakes(unsigned wakes) {
  ple_test_successful_empty_wakes = wakes;
}

void ple_fetcher_test_completion_on_last_wake(int enabled) {
  ple_test_completion_on_last_wake = !!enabled;
}
#endif
