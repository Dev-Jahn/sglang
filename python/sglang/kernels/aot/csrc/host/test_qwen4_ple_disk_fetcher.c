#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "qwen4_ple_disk_fetcher.h"

#define PAGE_BYTES 4096
#ifndef EUCLEAN
#define EUCLEAN 117
#endif

static int skip_errno(int error) {
  return error == EPERM || error == EACCES || error == ENOSYS || error == EOPNOTSUPP;
}

static int unavailable_result(void) {
  const char* required = getenv("SGL_KERNEL_PLE_REQUIRE_IO_URING");
  return required && strcmp(required, "1") == 0 ? 1 : 77;
}

static int run_scenarios(int file_fd, const unsigned char* page, int register_buffer) {
  int result = 1;
  void* buffer = NULL;
  void* fetcher = NULL;
  int allocation_error = posix_memalign(&buffer, PAGE_BYTES, 2 * PAGE_BYTES);
  if (allocation_error) {
    fprintf(stderr, "posix_memalign failed: %s\n", strerror(allocation_error));
    return 1;
  }
  if (ple_fetcher_abi_version() != PLE_FETCHER_ABI_VERSION ||
      ple_fetcher_lock_budget_ms() != PLE_FETCHER_LOCK_BUDGET_MS) {
    fprintf(stderr, "fetcher ABI or lock budget mismatch\n");
    goto done;
  }

  int failure_stage = 0;
  errno = 0;
  void* invalid_fetcher =
      ple_fetcher_create(file_fd, (unsigned char*)buffer + 1, 2 * PAGE_BYTES, 2, register_buffer, &failure_stage);
  if (invalid_fetcher || errno != EINVAL || failure_stage != PLE_FETCHER_FAILURE_NONE) {
    fprintf(stderr, "invalid create returned %p errno=%d stage=%d\n", invalid_fetcher, errno, failure_stage);
    goto done;
  }
  errno = 0;
  invalid_fetcher = ple_fetcher_create(-1, buffer, 2 * PAGE_BYTES, 2, register_buffer, &failure_stage);
  if (invalid_fetcher || errno != EINVAL || failure_stage != PLE_FETCHER_FAILURE_NONE) {
    fprintf(stderr, "invalid file descriptor returned %p errno=%d stage=%d\n", invalid_fetcher, errno, failure_stage);
    goto done;
  }
  fetcher = ple_fetcher_create(file_fd, buffer, 2 * PAGE_BYTES, 2, register_buffer, &failure_stage);
  if (!fetcher) {
    int error = errno;
    if (register_buffer && failure_stage == PLE_FETCHER_FAILURE_REGISTER_BUFFER &&
        (skip_errno(error) || error == ENOMEM)) {
      printf("PLE fetcher registered-buffer scenarios skipped: %s\n", strerror(error));
      result = unavailable_result();
      goto done;
    }
    if (!register_buffer && skip_errno(error)) {
      printf("PLE fetcher CTest skipped: io_uring is unavailable: %s\n", strerror(error));
      result = unavailable_result();
      goto done;
    }
    fprintf(
        stderr,
        "ple_fetcher_create(register_buffer=%d) failed at stage %d: %s\n",
        register_buffer,
        failure_stage,
        strerror(error));
    goto done;
  }

  void* deadline_fetcher = ple_fetcher_create(file_fd, buffer, 2 * PAGE_BYTES, 2, register_buffer, &failure_stage);
  if (!deadline_fetcher) {
    fprintf(stderr, "deadline test fetcher creation failed: %s\n", strerror(errno));
    goto done;
  }
  struct timespec deadline_start;
  struct timespec deadline_end;
  clock_gettime(CLOCK_MONOTONIC, &deadline_start);
  ple_fetcher_test_deadline_ms(20);
  ple_fetcher_test_interrupt_submissions(UINT_MAX);
  uint64_t deadline_offset = PAGE_BYTES;
  int deadline_rc = ple_fetcher_read(deadline_fetcher, &deadline_offset, 1, buffer, 2 * PAGE_BYTES);
  clock_gettime(CLOCK_MONOTONIC, &deadline_end);
  ple_fetcher_test_interrupt_submissions(0);
  ple_fetcher_test_deadline_ms(0);
  int64_t deadline_elapsed_ns = (int64_t)(deadline_end.tv_sec - deadline_start.tv_sec) * 1000000000LL +
                                (int64_t)(deadline_end.tv_nsec - deadline_start.tv_nsec);
  if (deadline_rc != -ETIMEDOUT || deadline_elapsed_ns < 0 || deadline_elapsed_ns > 500000000LL) {
    fprintf(
        stderr,
        "EINTR deadline returned %d after %llu ns\n",
        deadline_rc,
        (unsigned long long)(deadline_elapsed_ns < 0 ? 0 : deadline_elapsed_ns));
    ple_fetcher_destroy(deadline_fetcher);
    goto done;
  }
  deadline_rc = ple_fetcher_read(deadline_fetcher, &deadline_offset, 1, buffer, 2 * PAGE_BYTES);
  if (deadline_rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read after interrupted submission burst failed: %d\n", deadline_rc);
    goto done;
  }
  if (ple_fetcher_destroy(deadline_fetcher) != 0) {
    fprintf(stderr, "deadline test fetcher destroy failed\n");
    goto done;
  }

  uint64_t offsets[2] = {PAGE_BYTES, 2 * PAGE_BYTES};
  uint64_t misaligned_offset = PAGE_BYTES + 1;
  int rc = ple_fetcher_read(fetcher, &misaligned_offset, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EINVAL) {
    fprintf(stderr, "misaligned offset returned %d instead of %d\n", rc, -EINVAL);
    goto done;
  }
  rc = ple_fetcher_read(fetcher, offsets, 2, buffer, PAGE_BYTES);
  if (rc != -EFAULT) {
    fprintf(stderr, "short buffer returned %d instead of %d\n", rc, -EFAULT);
    goto done;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read check failed: %d\n", rc);
    goto done;
  }

  ple_fetcher_test_deadline_ms(20);
  ple_fetcher_test_expire_completion_deadline(1);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  ple_fetcher_test_deadline_ms(0);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "completion deadline returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read after completion deadline failed: %d\n", rc);
    goto done;
  }

  uint64_t invalid_offset = 3 * PAGE_BYTES;
  rc = ple_fetcher_read(fetcher, &invalid_offset, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EIO) {
    fprintf(stderr, "out-of-range read returned %d instead of %d\n", rc, -EIO);
    goto done;
  }
  unsigned failed_index = 99;
  int io_result = 99;
  rc = ple_fetcher_last_error(fetcher, &failed_index, &io_result);
  if (rc != 1 || failed_index != 0 || io_result != 0) {
    fprintf(stderr, "last_error returned rc=%d index=%u result=%d\n", rc, failed_index, io_result);
    goto done;
  }

  rc = ple_fetcher_read(fetcher, NULL, 0, buffer, 2 * PAGE_BYTES);
  if (rc != 0) {
    fprintf(stderr, "empty read returned %d\n", rc);
    goto done;
  }

  ple_fetcher_test_limit_submissions(1);
  rc = ple_fetcher_read(fetcher, offsets, 2, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0 || ple_fetcher_test_partial_submissions() == 0) {
    fprintf(stderr, "partial submission check failed: rc=%d partial=%u\n", rc, ple_fetcher_test_partial_submissions());
    goto done;
  }
  ple_fetcher_test_limit_submissions(0);

  ple_fetcher_test_successful_empty_wakes(3);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "successful wake accounting check failed: %d\n", rc);
    goto done;
  }

  /* The last allowed wake may make a completion visible. Reap it before
   * deciding that the no-progress budget is exhausted. */
  ple_fetcher_test_completion_on_last_wake(1);
  ple_fetcher_test_successful_empty_wakes(PLE_FETCHER_READ_WAITS);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "completion on final wake check failed: %d\n", rc);
    goto done;
  }

  unsigned timeouts = 0;
  for (unsigned index = 1; index < PLE_FETCHER_READ_WAITS; ++index) {
    if (!ple_fetcher_retry_after_timeout(&timeouts)) {
      fprintf(stderr, "timeout budget ended at %u\n", index);
      goto done;
    }
  }
  if (ple_fetcher_retry_after_timeout(&timeouts) ||
      (uint64_t)(timeouts + PLE_FETCHER_QUIESCE_WAITS) * PLE_FETCHER_WAIT_NS >
          (uint64_t)PLE_FETCHER_LOCK_BUDGET_MS * 1000000ULL) {
    fprintf(stderr, "timeout budget accounting failed: timeouts=%u\n", timeouts);
    goto done;
  }

  ple_fetcher_test_stall_wakes(PLE_FETCHER_READ_WAITS + 2);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "read-timeout quiesce returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read after timeout quiesce failed: %d\n", rc);
    goto done;
  }

  ple_fetcher_test_stall_wakes(PLE_FETCHER_READ_WAITS + 2);
  rc = ple_fetcher_read(fetcher, &invalid_offset, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "late-error quiesce returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  failed_index = 99;
  io_result = 99;
  rc = ple_fetcher_last_error(fetcher, &failed_index, &io_result);
  if (rc != 0) {
    fprintf(stderr, "quiesce replaced last_error: rc=%d index=%u result=%d\n", rc, failed_index, io_result);
    goto done;
  }

  void* accounting_fetcher = ple_fetcher_create(file_fd, buffer, 2 * PAGE_BYTES, 2, register_buffer, &failure_stage);
  if (!accounting_fetcher) {
    fprintf(stderr, "accounting test fetcher creation failed: %s\n", strerror(errno));
    goto done;
  }
  ple_fetcher_test_corrupt_accounting_once();
  rc = ple_fetcher_read(accounting_fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "ring accounting corruption returned %d instead of %d\n", rc, -EUCLEAN);
    ple_fetcher_destroy(accounting_fetcher);
    goto done;
  }
  if (ple_fetcher_destroy(accounting_fetcher) != 0) {
    fprintf(stderr, "accounting test fetcher destroy failed\n");
    goto done;
  }

  ple_fetcher_test_stall_completions(1);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "poison setup returned %d instead of %d\n", rc, -EUCLEAN);
    goto done;
  }
  ple_fetcher_test_stall_completions(0);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "poisoned fetcher returned %d instead of %d\n", rc, -EUCLEAN);
    goto done;
  }
  if (!ple_fetcher_test_ring_open(fetcher)) {
    fprintf(stderr, "poisoned fetcher closed its ring before destroy\n");
    goto done;
  }
  rc = ple_fetcher_read(fetcher, NULL, 0, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "poisoned empty read returned %d instead of %d\n", rc, -EUCLEAN);
    goto done;
  }

  ple_fetcher_test_stall_completions(1);
  ple_fetcher_test_deadline_ms(20);
  ple_fetcher_test_interrupt_waits(UINT_MAX);
  clock_gettime(CLOCK_MONOTONIC, &deadline_start);
  rc = ple_fetcher_destroy(fetcher);
  clock_gettime(CLOCK_MONOTONIC, &deadline_end);
  ple_fetcher_test_interrupt_waits(0);
  ple_fetcher_test_deadline_ms(0);
  deadline_elapsed_ns = (int64_t)(deadline_end.tv_sec - deadline_start.tv_sec) * 1000000000LL +
                        (int64_t)(deadline_end.tv_nsec - deadline_start.tv_nsec);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "destroy timeout returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  if (deadline_elapsed_ns < 0 || deadline_elapsed_ns > 500000000LL) {
    fprintf(stderr, "destroy EINTR bound took %llu ns\n", (unsigned long long)deadline_elapsed_ns);
    goto done;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "read after destroy timeout returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  rc = ple_fetcher_last_error(fetcher, &failed_index, &io_result);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "last_error after destroy timeout returned %d instead of %d\n", rc, -ETIMEDOUT);
    goto done;
  }
  rc = ple_fetcher_destroy(fetcher);
  if (rc != -ETIMEDOUT || !ple_fetcher_test_ring_open(fetcher)) {
    fprintf(stderr, "destroy retry returned %d or released retained mappings\n", rc);
    goto done;
  }

  /* The terminal fetcher and its staging buffer remain live until exit. */
  fetcher = NULL;
  buffer = NULL;

  result = 0;
done:
  ple_fetcher_test_limit_submissions(0);
  ple_fetcher_test_stall_completions(0);
  ple_fetcher_test_stall_wakes(0);
  ple_fetcher_test_successful_empty_wakes(0);
  ple_fetcher_test_completion_on_last_wake(0);
  ple_fetcher_test_interrupt_submissions(0);
  ple_fetcher_test_interrupt_waits(0);
  ple_fetcher_test_deadline_ms(0);
  ple_fetcher_test_expire_completion_deadline(0);
  if (fetcher) {
    int destroy_rc = ple_fetcher_destroy(fetcher);
    if (destroy_rc && result == 0) {
      fprintf(stderr, "ple_fetcher_destroy returned %d\n", destroy_rc);
      result = 1;
    }
  }
  free(buffer);
  return result;
}

int main(void) {
  const char* temp_dir = getenv("TMPDIR");
  if (!temp_dir || !*temp_dir) temp_dir = "/tmp";
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/qwen4-ple-fetcher-XXXXXX", temp_dir) >= (int)sizeof(path)) {
    fprintf(stderr, "image directory path is too long\n");
    return 1;
  }
  int buffered_fd = mkstemp(path);
  if (buffered_fd < 0) {
    perror("mkstemp");
    return 1;
  }
  unsigned char page[PAGE_BYTES];
  for (size_t index = 0; index < sizeof(page); ++index)
    page[index] = (unsigned char)(index & 255);
  if (ftruncate(buffered_fd, 3 * PAGE_BYTES) < 0 || pwrite(buffered_fd, page, PAGE_BYTES, PAGE_BYTES) != PAGE_BYTES) {
    perror("prepare temp file");
    close(buffered_fd);
    unlink(path);
    return 1;
  }
  close(buffered_fd);

  int file_fd = open(path, O_RDONLY | O_DIRECT);
  unlink(path);
  if (file_fd < 0) {
    if (errno == EINVAL) {
      fprintf(
          stderr, "O_DIRECT is unavailable in TMPDIR=%s; TMPDIR must point to a block-backed filesystem\n", temp_dir);
      return 1;
    }
    if (skip_errno(errno)) {
      printf("PLE fetcher CTest skipped: O_DIRECT is unavailable: %s\n", strerror(errno));
      return unavailable_result();
    }
    perror("open O_DIRECT");
    return 1;
  }

  int result = run_scenarios(file_fd, page, 0);
  if (result == 0) {
    int registered_result = run_scenarios(file_fd, page, 1);
    if (registered_result == 77) {
      printf("PLE fetcher unregistered-buffer scenarios passed; registered-buffer scenarios skipped\n");
    } else {
      result = registered_result;
    }
  }
  close(file_fd);
  return result;
}
