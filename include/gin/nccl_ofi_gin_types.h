/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_TYPES_H
#define NCCL_OFI_GIN_TYPES_H

#include <stdint.h>

/**
 * Forward-declarations of GIN types
 */
class nccl_ofi_gin_mr_handle_t;
class nccl_ofi_gin_comm;
class nccl_ofi_gin_resources;
struct nccl_ofi_gin_ep_rail_t;

/**
 * Represents metadata associated with a put-signal request. This is sent from
 * the put-signal initiator to the target.
 */
struct nccl_net_ofi_gin_signal_metadata_msg_t {
	/* Signal information (if applicable) */
	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;

	/* Message sequence number */
	uint32_t msg_seq_num;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id;

	/* Number of completions the target will receive
	 *
	 * This will be either 1 or 2
	 * 1: if this is a signal without any associated data (zero-sized
	 *    put-signal) or data without any signal (put)
	 * 2: For put-signal (data + signal) */
	uint32_t num_segments;
};

/**
 * Constants
 */
#define MAX_NUM_RAILS (4)

/**
 * Immediate data format (32 bits).
 *
 * Bit 0 distinguishes message type: 0 = non-ACK, 1 = ACK.
 * Bits 1-10 (comm_id) and bits 11-21 (seq_num) are common to both formats.
 *
 * Non-ACK (bit 0 = 0):
 *
 *   31      27  26       22    21       11  10       1   0
 *  [unused  ] [ack_req] [seg_cnt ] [msg_seq_num] [comm_id ] [ 0 ]
 *   5 bits     1 bit     4 bits     11 bits       10 bits    1b
 *
 * ACK (bit 0 = 1):
 *
 *   31      22  21       11  10       1   0
 *  [ack_count] [ack_seq_num] [comm_id ] [ 1 ]
 *   10 bits     11 bits       10 bits    1b
 *
 * comm_id is 10 bits (1024 values). At most NCCL_GIN_MAX_CONTEXTS (4)
 * gin_comms share an endpoint, so at most 4 comm_id values are in use
 * per endpoint at any time. 10 bits provides ample headroom.
 *
 * ack_seq_num is the high-water mark of the ACK range. ack_count is the
 * number of seq_nums in the range; the sender computes
 * start_seq = (ack_seq_num - count + 1) & SEQ_MASK. The count is bounded
 * by the GFD queue depth (NCCL_NET_MAX_REQUESTS = 32) which caps in-flight
 * seq_nums per peer due to head-of-line blocking at cis.
 *
 * Both ack_seq_num and count are needed so each ACK is self-contained.
 * A simpler scheme using only ack_seq_num would require the sender to track
 * last_acked_seq_num and clear from there to ack_seq_num. This breaks because
 * EFA does not guarantee fi_writedata completion ordering: a stale ACK
 * (e.g. ack_seq_num=0) arriving after a newer ACK (e.g. ack_seq_num=5) causes
 * the sender to walk from last_acked_seq_num=6 forward through the entire
 * sequence space back to 0, clearing in-flight requests that have not
 * been delivered — corrupting data. By encoding both ack_seq_num and count,
 * each ACK describes an independent range and no cumulative state is needed
 * on the sender, so reordered ACKs are harmless.
 */

/* Bit 0: message type. 0 = non-ACK, 1 = ACK */
#define GIN_IMM_IS_ACK(data)   ((data) & 1)

/* Common fields (same position in both formats) */
#define GIN_IMM_COMM_SHIFT     1
#define GIN_IMM_COMM_BITS      10
#define GIN_IMM_SEQ_SHIFT      11
#define GIN_IMM_SEQ_BITS       11
#define GIN_IMM_SEQ_MASK       ((1 << GIN_IMM_SEQ_BITS) - 1)
#define GIN_IMM_COMM_MASK      ((1 << GIN_IMM_COMM_BITS) - 1)
#define GIN_MAX_COMMS          (1 << GIN_IMM_COMM_BITS)

#define GIN_IMM_GET_COMM_ID(data)  (((data) >> GIN_IMM_COMM_SHIFT) & GIN_IMM_COMM_MASK)
#define GIN_IMM_GET_SEQ_NUM(data)  (((data) >> GIN_IMM_SEQ_SHIFT) & GIN_IMM_SEQ_MASK)

/* Non-ACK: upper bits = [5 unused | 1 ack_req | 4 seg_count] */
#define GIN_IMM_GET_SEG_CNT(data)       (((data) >> 22) & 0xF)
#define GIN_IMM_GET_ACK_REQUESTED(data) (((data) >> 26) & 1)
#define GIN_IMM_SEG_DATA(comm_id, seq, nseg, ack_req)                                               \
	(((ack_req) << 26) | ((nseg) << 22) |                                                      \
	 ((seq) << GIN_IMM_SEQ_SHIFT) | ((comm_id) << GIN_IMM_COMM_SHIFT))

/* ACK: upper 10 bits = ack_count, bit 0 = 1 */
#define GIN_IMM_ACK_GET_COUNT(data) (((data) >> 22) & 0x3FF)
#define GIN_IMM_ACK_DATA(comm_id, seq, count)                                                      \
	(((count) << 22) | ((seq) << GIN_IMM_SEQ_SHIFT) |                                          \
	 ((comm_id) << GIN_IMM_COMM_SHIFT) | 1)

/* ACK interval for PUT-only messages. Send an ACK every N consecutive PUTs
   to prevent sequence number wraparound. */
#define GIN_ACK_INTERVAL 64
static_assert(GIN_ACK_INTERVAL <= (1 << (GIN_IMM_SEQ_BITS - 1)),
	      "GIN_ACK_INTERVAL must not exceed half the sequence number space");

#endif
