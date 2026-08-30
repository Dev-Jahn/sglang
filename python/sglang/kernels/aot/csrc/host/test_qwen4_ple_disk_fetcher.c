#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "qwen4_ple_disk_fetcher.h"

#define PAGE_BYTES 4096

#ifndef EUCLEAN
#define EUCLEAN 117
#endif

static int skip_errno(int error) {
  return error == EPERM || error == EACCES || error == ENOSYS || error == EOPNOTSUPP || error == ENOMEM;
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
    if (skip_errno(errno) || errno == EINVAL) {
      printf("PLE fetcher CTest skipped: O_DIRECT is unavailable: %s\n", strerror(errno));
      return 77;
    }
    perror("open O_DIRECT");
    return 1;
  }

  void* buffer = NULL;
  if (posix_memalign(&buffer, PAGE_BYTES, 2 * PAGE_BYTES) != 0) {
    close(file_fd);
    return 1;
  }
  int failure_stage = 0;
  void* fetcher = ple_fetcher_create(file_fd, buffer, 2 * PAGE_BYTES, 2, 0, &failure_stage);
  if (!fetcher) {
    int error = errno;
    free(buffer);
    close(file_fd);
    if (skip_errno(error)) {
      printf("PLE fetcher CTest skipped: io_uring is unavailable: %s\n", strerror(error));
      return 77;
    }
    fprintf(stderr, "ple_fetcher_create failed at stage %d: %s\n", failure_stage, strerror(error));
    return 1;
  }

  uint64_t offsets[2] = {PAGE_BYTES, 2 * PAGE_BYTES};
  int rc = ple_fetcher_read(fetcher, offsets, 2, buffer, PAGE_BYTES);
  if (rc != -EFAULT) {
    fprintf(stderr, "short buffer returned %d instead of %d\n", rc, -EFAULT);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read check failed: %d\n", rc);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  rc = ple_fetcher_read(fetcher, NULL, 0, buffer, 2 * PAGE_BYTES);
  if (rc != 0) {
    fprintf(stderr, "empty read returned %d\n", rc);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  ple_fetcher_test_limit_submissions(1);
  rc = ple_fetcher_read(fetcher, offsets, 2, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0 || ple_fetcher_test_partial_submissions() == 0) {
    fprintf(stderr, "partial submission check failed: rc=%d partial=%u\n", rc, ple_fetcher_test_partial_submissions());
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }
  ple_fetcher_test_limit_submissions(0);

  ple_fetcher_test_successful_empty_wakes(3);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "successful wake accounting check failed: %d\n", rc);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  /* The last allowed wake may make a completion visible. Reap it before
   * deciding that the no-progress budget is exhausted. */
  ple_fetcher_test_completion_on_last_wake(1);
  ple_fetcher_test_successful_empty_wakes(PLE_FETCHER_MAX_WAITS);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "completion on final wake check failed: %d\n", rc);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  unsigned timeouts = 0;
  for (unsigned index = 1; index < PLE_FETCHER_MAX_WAITS; ++index) {
    if (!ple_fetcher_retry_after_timeout(&timeouts)) {
      fprintf(stderr, "timeout budget ended at %u\n", index);
      ple_fetcher_destroy(fetcher);
      free(buffer);
      close(file_fd);
      return 1;
    }
  }
  if (ple_fetcher_retry_after_timeout(&timeouts) || (uint64_t)timeouts * PLE_FETCHER_WAIT_NS != 5000000000ULL) {
    fprintf(stderr, "timeout budget accounting failed: timeouts=%u\n", timeouts);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  ple_fetcher_test_stall_wakes(PLE_FETCHER_MAX_WAITS + 2);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -ETIMEDOUT) {
    fprintf(stderr, "read-timeout quiesce returned %d instead of %d\n", rc, -ETIMEDOUT);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != 0 || memcmp(buffer, page, PAGE_BYTES) != 0) {
    fprintf(stderr, "read after timeout quiesce failed: %d\n", rc);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  ple_fetcher_test_stall_completions(1);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "poison setup returned %d instead of %d\n", rc, -EUCLEAN);
    ple_fetcher_test_stall_completions(0);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }
  ple_fetcher_test_stall_completions(0);
  rc = ple_fetcher_read(fetcher, offsets, 1, buffer, 2 * PAGE_BYTES);
  if (rc != -EUCLEAN) {
    fprintf(stderr, "poisoned fetcher returned %d instead of %d\n", rc, -EUCLEAN);
    ple_fetcher_destroy(fetcher);
    free(buffer);
    close(file_fd);
    return 1;
  }

  ple_fetcher_destroy(fetcher);
  free(buffer);
  close(file_fd);
  return 0;
}
