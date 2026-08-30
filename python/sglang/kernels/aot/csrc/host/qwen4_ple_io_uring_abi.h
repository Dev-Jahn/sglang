/* SPDX-License-Identifier: Apache-2.0 */
/* ABI definitions follow Linux UAPI include/uapi/linux/io_uring.h. */
#ifndef QWEN4_PLE_IO_URING_ABI_H
#define QWEN4_PLE_IO_URING_ABI_H

#include <stdint.h>
#include <sys/syscall.h>

#if defined(__x86_64__) || defined(__aarch64__)
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif
#else
#if !defined(__NR_io_uring_setup) || !defined(__NR_io_uring_enter) || !defined(__NR_io_uring_register)
#error "io_uring syscall numbers are unavailable for this architecture"
#endif
#endif

struct io_uring_sqe {
  uint8_t opcode;
  uint8_t flags;
  uint16_t ioprio;
  int32_t fd;
  uint64_t off;
  uint64_t addr;
  uint32_t len;
  union {
    uint32_t rw_flags;
    uint32_t cancel_flags;
  };
  uint64_t user_data;
  union {
    uint16_t buf_index;
    uint16_t buf_group;
  } __attribute__((packed));
  uint16_t personality;
  union {
    int32_t splice_fd_in;
    uint32_t file_index;
  };
  uint64_t addr3;
  uint64_t pad2;
};

struct io_uring_cqe {
  uint64_t user_data;
  int32_t res;
  uint32_t flags;
};

struct io_sqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t flags;
  uint32_t dropped;
  uint32_t array;
  uint32_t resv1;
  uint64_t user_addr;
};

struct io_cqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t overflow;
  uint32_t cqes;
  uint32_t flags;
  uint32_t resv1;
  uint64_t user_addr;
};

struct io_uring_params {
  uint32_t sq_entries;
  uint32_t cq_entries;
  uint32_t flags;
  uint32_t sq_thread_cpu;
  uint32_t sq_thread_idle;
  uint32_t features;
  uint32_t wq_fd;
  uint32_t resv[3];
  struct io_sqring_offsets sq_off;
  struct io_cqring_offsets cq_off;
};

struct io_uring_getevents_arg {
  uint64_t sigmask;
  uint32_t sigmask_sz;
  uint32_t pad;
  uint64_t ts;
};

struct qwen4_ple_kernel_timespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

enum {
  IORING_OP_READ_FIXED = 4,
  IORING_OP_READ = 22,
};

#define IOSQE_FIXED_FILE (1U << 0)
#define IORING_OFF_SQ_RING 0ULL
#define IORING_OFF_CQ_RING 0x8000000ULL
#define IORING_OFF_SQES 0x10000000ULL
#define IORING_ENTER_GETEVENTS (1U << 0)
#define IORING_ENTER_EXT_ARG (1U << 3)
#define IORING_FEAT_SINGLE_MMAP (1U << 0)
#define IORING_FEAT_EXT_ARG (1U << 8)

enum {
  IORING_REGISTER_BUFFERS = 0,
  IORING_UNREGISTER_BUFFERS = 1,
  IORING_REGISTER_FILES = 2,
  IORING_UNREGISTER_FILES = 3,
};

#endif
