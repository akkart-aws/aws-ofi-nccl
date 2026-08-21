/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path. See
 * nccl_ofi_gin_gdaki_resources.h for the public declarations.
 */

#include "config.h"

#include <bit>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"

#include "efa_cuda_dp.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

/* Plain-C CQE layout from efa-dp-direct (no CUDA tokens; includable here). */
#include "efa_io_defs.h"

/* efa_io_defs.h declares its field masks in the .cuh alongside __device__ accessors.
 * Decode the one flag byte we need directly; the layout is fixed by the device ABI
 * (efa_io_cdesc_common::flags): bit 0 : phase. */
#define GDAKI_CDESC_FLAG_PHASE(flags)  ((int)((flags) & 0x1u))

const char *nccl_ofi_gin_gdaki_comp_status_str(uint8_t status)
{
	switch (status) {
	case EFA_IO_COMP_STATUS_OK: return "OK";
	case EFA_IO_COMP_STATUS_FLUSHED: return "FLUSHED (QP destroyed)";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR: return "LOCAL_ERROR_QP_INTERNAL_ERROR";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP: return "LOCAL_ERROR_UNSUPPORTED_OP";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH: return "LOCAL_ERROR_INVALID_AH";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY: return "LOCAL_ERROR_INVALID_LKEY";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH: return "LOCAL_ERROR_BAD_LENGTH";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS: return "REMOTE_ERROR_BAD_ADDRESS (bad rkey/IOVA)";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT: return "REMOTE_ERROR_ABORT";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN: return "REMOTE_ERROR_BAD_DEST_QPN";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR: return "REMOTE_ERROR_RNR";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH: return "REMOTE_ERROR_BAD_LENGTH";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS: return "REMOTE_ERROR_BAD_STATUS";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE: return "LOCAL_ERROR_UNRESP_REMOTE";
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER: return "REMOTE_ERROR_UNKNOWN_PEER (no AH at remote)";
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE: return "LOCAL_ERROR_UNREACH_REMOTE";
	default: return "UNKNOWN";
	}
}

/*
 * Build fi_getinfo hints for the GDAKI endpoint.
 *
 * GDAKI states its own requirements explicitly (rather than copying
 * ref_info's attributes); ref_info is used only for the fabric /
 * domain / provider names that narrow fi_getinfo to the single
 * efa-direct provider entry the proxy already opened. This mirrors
 * the proxy plugin's get_gin_hints pattern in nccl_ofi_gin_resources.cpp.
 *
 * GDAKI does not register memory on this EP — the proxy's regMrSym
 * registers on the shared domain — and does not do fi_cq_readfrom,
 * so FI_SOURCE is not requested. FI_HMEM is still needed because the endpoint
 * is used to access GPU memory. efa-direct requires FI_CONTEXT2 per fi_efa(7).
 */
static void get_gdaki_hints(struct fi_info &hints, struct fi_info *ref_info)
{
	hints.caps = FI_MSG | FI_RMA | FI_HMEM;
	hints.mode = FI_CONTEXT2;

	hints.ep_attr->type = FI_EP_RDM;
	hints.addr_format = FI_ADDR_EFA;

	hints.domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM |
				     FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
				     FI_MR_PROV_KEY;
	hints.domain_attr->threading = FI_THREAD_SAFE;
	hints.domain_attr->control_progress = FI_PROGRESS_AUTO;
	hints.domain_attr->data_progress = FI_PROGRESS_AUTO;

	/* Narrow fi_getinfo to the provider / fabric / domain the proxy
	 * already opened. Names are required to obtain exactly one result. */
	hints.fabric_attr->prov_name = strdup(ref_info->fabric_attr->prov_name);
	hints.fabric_attr->name = strdup(ref_info->fabric_attr->name);
	hints.domain_attr->name = strdup(ref_info->domain_attr->name);
}

/*
 * Obtain a GDAKI-owned fi_info via fi_getinfo, narrowed to exactly the
 * fabric / domain the proxy reference points at.
 */
static struct fi_info *get_gdaki_info(struct fi_info *ref_info)
{
	struct fi_info *hints = fi_allocinfo();
	if (hints == nullptr) {
		throw std::runtime_error("fi_allocinfo for GDAKI hints failed");
	}
	get_gdaki_hints(*hints, ref_info);

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL,
			     hints, &results);
	fi_freeinfo(hints);
	if (ret != 0) {
		throw std::runtime_error("fi_getinfo for GDAKI info failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (results == nullptr) {
		throw std::runtime_error("fi_getinfo returned no GDAKI providers");
	}
	if (results->next != nullptr) {
		fi_freeinfo(results);
		throw std::runtime_error(
			"fi_getinfo returned more than one GDAKI provider; "
			"hints were not narrow enough");
	}

	return results;
}

void gdaki_fi_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     struct fid_cq *cq)
{
	if (ep || av || info) {
		throw std::runtime_error("gdaki_fi_endpoint: double open");
	}

	info = get_gdaki_info(ref_info);

	struct fi_av_attr av_attr = {};
	av_attr.type = FI_AV_TABLE;
	int ret = fi_av_open(domain, &av_attr, &av, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_av_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_endpoint(domain, info, &ep, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_endpoint on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	/* Bind the borrowed cq; the caller owns and closes it. */
	ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind CQ failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind AV failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::enable()
{
	int ret = fi_enable(ep);
	if (ret != 0) {
		throw std::runtime_error("fi_enable failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::bind(struct fid *fid, uint64_t flags)
{
	int ret = fi_ep_bind(ep, fid, flags);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

gdaki_gpu_qp::~gdaki_gpu_qp()
{
	if (qp != nullptr) {
		efa_cuda_destroy_qp(qp);
	}
}

void gdaki_gpu_qp::build(const struct fi_efa_wq_attr &sq_attr,
			 const struct fi_efa_wq_attr &rq_attr,
			 void *sq_buf_dev, void *sq_db_dev)
{
	/* The host API allocates GPU memory; rebuilding would overwrite the only
	 * owned pointer and leak the previous descriptor. */
	if (qp != nullptr) {
		throw std::runtime_error("gdaki_gpu_qp: double build");
	}

	/* The plugin supplies provider-probed buffers and geometry. The host API
	 * validates the attributes and initializes queue masks, counters, and phase
	 * state before uploading the canonical descriptor to GPU memory. */
	struct efa_cuda_qp_attrs attrs = {};
	attrs.sq_buffer = static_cast<uint8_t *>(sq_buf_dev);
	attrs.rq_buffer = static_cast<uint8_t *>(rq_attr.buffer);
	attrs.sq_doorbell = static_cast<uint32_t *>(sq_db_dev);
	attrs.rq_doorbell = static_cast<uint32_t *>(rq_attr.doorbell);
	attrs.sq_num_entries = sq_attr.num_entries;
	attrs.sq_entry_size = sq_attr.entry_size;
	attrs.sq_max_batch = sq_attr.max_batch;
	attrs.rq_num_entries = rq_attr.num_entries;
	attrs.rq_entry_size = rq_attr.entry_size;

	qp = efa_cuda_create_qp(&attrs, sizeof(attrs));
	if (qp == nullptr) {
		throw std::runtime_error("gdaki_gpu_qp: efa_cuda_create_qp failed");
	}
}

void gdaki_host_cq::open(struct fid_domain *domain, size_t cq_size)
{
	if (cq_ != nullptr) {
		throw std::runtime_error("gdaki_host_cq: double open");
	}
	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = cq_size;
	int ret = fi_cq_open(domain, &cq_attr, &cq_, nullptr);
	if (ret != 0) {
		throw std::runtime_error("gdaki_host_cq: fi_cq_open failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_host_cq::build(struct fi_efa_ops_gda *gda_ops, int ctx_id)
{
	if (cq_ == nullptr) {
		throw std::runtime_error("gdaki_host_cq: build before open");
	}

	struct fi_efa_cq_attr efa_cq_attr = {};
	int ret = gda_ops->query_cq(cq_, &efa_cq_attr);
	if (ret != 0) {
		throw std::runtime_error("gdaki_host_cq: query_cq failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (efa_cq_attr.buffer == nullptr || efa_cq_attr.entry_size == 0 ||
	    efa_cq_attr.num_entries == 0) {
		throw std::runtime_error("gdaki_host_cq: invalid CQ geometry for ctx" + std::to_string(ctx_id));
	}
	if ((efa_cq_attr.num_entries & (efa_cq_attr.num_entries - 1)) != 0) {
		throw std::runtime_error("gdaki_host_cq: CQ num_entries not power of two for ctx" + std::to_string(ctx_id));
	}

	cq_buf = static_cast<const uint8_t *>(efa_cq_attr.buffer);
	entry_size = efa_cq_attr.entry_size;
	num_entries = efa_cq_attr.num_entries;
	cq_mask = num_entries - 1;
	uint32_t shift = 0;
	while ((1u << shift) < num_entries) {
		++shift;
	}
	cq_size_shift = shift;
	cc = 0;
	phase = 1;   /* The phase starts at 1 to match efa_cuda_create_cq's initial phase. */
}


/*
 * Sentinel for "this peer has no endpoint at this slot". A rank that did
 * not create an endpoint for a given (collective) allgather round leaves
 * its slot zero-filled; ranks are NOT required to request the same number
 * of signal/counter endpoints, so some (slot, peer) pairs have no remote
 * endpoint. A real EFA address (FI_ADDR_EFA: AH + QPN + QKEY-derived bytes)
 * is never all-zero, so an all-zero slot unambiguously means "absent" and
 * is skipped during addressing — the corresponding table entry stays 0 and
 * is never used by a correct caller (the kernel bounds each post by the
 * target's advertised count).
 */
static bool gdaki_ep_addr_is_absent(const uint8_t *addr, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		if (addr[i] != 0) {
			return false;
		}
	}
	return true;
}

void gdaki_target_addressing::populate(gdaki_fi_endpoint &endpoint,
				       const std::vector<uint8_t> &all_addrs,
				       size_t ep_addr_len, int total_slots, int nranks,
				       struct fi_efa_ops_gda *gda_ops)
{
	if (total_slots <= 0 || nranks <= 0) {
		return;
	}

	const size_t n = (size_t)total_slots * (size_t)nranks;
	ahs.allocate(n);
	qpns.allocate(n);
	qkeys.allocate(n);

	/* Build the targetSlot-major table: idx = slot * nranks + peer.
	 *
	 * `all_addrs` is the batched-allgather buffer, peer-major:
	 *     addr(peer, slot) = &all_addrs[(peer * total_slots + slot) * ep_addr_len]
	 * with slot 0 = peer's data EP and slots 1..(total_slots-1) = peer's
	 * sc EPs. We resolve every (slot, peer) through THIS endpoint's own AV
	 * (an address handle is AV-local) and store it at [slot * nranks + peer].
	 *
	 * A (slot, peer) whose peer has no endpoint at that slot (asymmetric
	 * counts leave a zero address) is skipped; the entry stays 0 and is
	 * never addressed — the kernel bounds each signalling post by the
	 * target peer's advertised signal count. */
	for (int slot = 0; slot < total_slots; slot++) {
		for (int peer = 0; peer < nranks; peer++) {
			const size_t dst_idx = (size_t)slot * (size_t)nranks + (size_t)peer;
			const size_t src_idx = (size_t)peer * (size_t)total_slots + (size_t)slot;
			const uint8_t *addr = all_addrs.data() + src_idx * ep_addr_len;

			if (gdaki_ep_addr_is_absent(addr, ep_addr_len)) {
				continue;
			}

			fi_addr_t fi_addr;
			int ret = fi_av_insert(endpoint.av, addr, 1, &fi_addr, 0, nullptr);
			if (ret != 1) {
				throw std::runtime_error(
					"target fi_av_insert failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			uint16_t ahn = 0, remote_qpn = 0;
			uint32_t remote_qkey = 0;
			ret = gda_ops->query_addr(endpoint.ep, fi_addr, &ahn,
						  &remote_qpn, &remote_qkey);
			if (ret != 0) {
				throw std::runtime_error(
					"target query_addr failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			ahs.host[dst_idx] = ahn;
			qpns.host[dst_idx] = remote_qpn;
			qkeys.host[dst_idx] = remote_qkey;
		}
	}

	ahs.commit();
	qpns.commit();
	qkeys.commit();
}

void gdaki_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			  struct fid_cq *cq)
{
	endpoint.open(domain, ref_info, cq);
	endpoint.enable();
}

void gdaki_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
			      const std::vector<uint8_t> &all_addrs,
			      size_t ep_addr_len, int total_slots, int nranks)
{
	/* Query QP and map SQ MMIO for GPU access. */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	int ret = gda_ops->query_qp_wqs(endpoint.ep, &sq_attr, &rq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_qp_wqs failed: " +
					 std::string(fi_strerror(-ret)));

	sq_buffer.map(sq_attr.buffer,
		      (size_t)sq_attr.num_entries * sq_attr.entry_size);

	/* rdma-core mmaps the doorbell MMIO region with sysconf(_SC_PAGESIZE)
	 * (see providers/efa/verbs.c). Use the plugin's cached system_page_size
	 * so our GPU-side mapping covers the same region rdma-core opened. */
	sq_doorbell.map(sq_attr.doorbell, system_page_size);

	gpu_qp.build(sq_attr, rq_attr, sq_buffer.dev, sq_doorbell.dev);

	/* Stash SQ ring depth for the device-side SQ-overflow backpressure check. */
	sq_size = sq_attr.num_entries;

	/* Build the [total_slots*nranks] target table in GPU memory. */
	targets.populate(endpoint, all_addrs, ep_addr_len, total_slots, nranks, gda_ops);
}

void gdaki_data_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			       struct fi_efa_ops_gda *gda_ops, struct fid_cq *cq)
{
	/* Create the FI_WRITE counter first; it is bound to the inner endpoint
	 * between open() and enable() and is this QP's per-QP completion source
	 * (SQ ring reuse + blocking Flush). */
	write_cntr.create(gda_ops, domain);

	base.endpoint.open(domain, ref_info, cq);
	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);
	base.endpoint.enable();
}

void gdaki_data_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
				   const std::vector<uint8_t> &all_addrs,
				   size_t ep_addr_len, int total_slots, int nranks)
{
	base.populate(gda_ops, all_addrs, ep_addr_len, total_slots, nranks);
	/* Per-QP completion is this endpoint's FI_WRITE NIC counter. */
	base.completed_count_dev = write_cntr.gpu_ptr();
}

void gdaki_sc_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     struct fi_efa_ops_gda *gda_ops, struct fid_cq *cq)
{
	/* Create hardware counters first; they will be bound to the inner
	 * endpoint between open() and enable(). */
	write_cntr.create(gda_ops, domain);
	remote_write_cntr.create(gda_ops, domain);

	/* Open the inner endpoint on the context's shared CQ, without enable. */
	base.endpoint.open(domain, ref_info, cq);

	/* Bind counters before enabling. */
	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);
	base.endpoint.bind(&remote_write_cntr.get()->fid, FI_REMOTE_WRITE);

	base.endpoint.enable();
}

void gdaki_sc_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
				 const std::vector<uint8_t> &all_addrs,
				 size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP query, MMIO map, GPU descriptors,
	 * target table) to the inner endpoint. */
	base.populate(gda_ops, all_addrs, ep_addr_len, total_slots, nranks);
	/* Per-QP completion is this endpoint's FI_WRITE NIC counter (same as data). */
	base.completed_count_dev = write_cntr.gpu_ptr();

	/*
	 * Build the two device handles. They share QP / CQ / target addressing /
	 * sq_size / submitted_count / completed_count layout — only cntr_value
	 * differs. Per-QP completion is the FI_WRITE counter (completed_count); the
	 * counters below are the user-facing counter/signal values the kernel reads.
	 *
	 * - counter_dev_handle exposes the FI_WRITE counter via cntr_value (local
	 *   write count). Returned to the kernel through counter_handles[].
	 * - signal_dev_handle exposes the FI_REMOTE_WRITE counter via cntr_value
	 *   (signal arrivals). Returned to the kernel through signal_handles[].
	 *
	 * All pointers are set on the host before commit() pushes the struct to GPU
	 * memory.
	 */
	auto fill_common = [&](nccl_ofi_gin_gdaki_dev_counter_handle &h) {
		h.base.qp = base.gpu_qp.dev();
		h.base.target_address_handles = base.targets.ahs.dev;
		h.base.target_remote_qpns = base.targets.qpns.dev;
		h.base.target_qkey = base.targets.qkeys.dev;
		h.base.submitted_count = 0;
		h.base.completed_count = base.completed_count_dev;
		h.base.sq_size = base.sq_size;
		h.cntr_offset = 0;   /* offset-based reset baseline */
	};

	counter_dev_handle.allocate(1);
	fill_common(counter_dev_handle.host[0]);
	/* TODO: Refactor counter_dev_handle so the same gpu memory is not
	 * being used by multiple fields */
	counter_dev_handle.host[0].cntr_value = write_cntr.gpu_ptr();
	counter_dev_handle.commit();

	signal_dev_handle.allocate(1);
	fill_common(signal_dev_handle.host[0]);
	signal_dev_handle.host[0].cntr_value = remote_write_cntr.gpu_ptr();
	signal_dev_handle.commit();
}

gdaki_completion_state::~gdaki_completion_state()
{
	if (completions_table_reg != nullptr) {
		get_device_copy().deregister_region(completions_table_reg);
		completions_table_reg = nullptr;
	}
	if (completions_table_dev != nullptr) {
		nccl_net_ofi_gpu_mem_free(completions_table_dev);
		completions_table_dev = nullptr;
	}
}

void gdaki_completion_state::allocate(int nContexts_in, int nranks_in)
{
	if (completions_table_dev != nullptr) {
		throw std::runtime_error("gdaki_completion_state: allocate called twice");
	}
	if (nContexts_in <= 0 || nranks_in <= 0) {
		throw std::runtime_error("gdaki_completion_state: allocate with zero contexts/ranks");
	}
	nContexts = nContexts_in;
	nranks = nranks_in;

	/* Completions table words: per-context completed [nContexts], then
	 * peer_completion [nContexts * nranks] (8 bytes = 1 word each). Per-QP
	 * completion lives in each endpoint's FI_WRITE NIC counter. */
	ctx_word_base = 0;
	peer_word_base = (size_t)nContexts;
	completions_table_words = peer_word_base + (size_t)nContexts * (size_t)nranks;

	void *dev = nullptr;
	if (nccl_net_ofi_gpu_mem_alloc(&dev, completions_table_words * sizeof(uint64_t)) != 0) {
		throw std::runtime_error("gdaki_completion_state: gpu_mem_alloc failed");
	}
	completions_table_dev = static_cast<uint64_t *>(dev);

	nccl_ofi_device_copy::RegHandle *reg = nullptr;
	if (get_device_copy().register_region(completions_table_dev, completions_table_words * sizeof(uint64_t), reg) != 0) {
		throw std::runtime_error("gdaki_completion_state: gdrcopy register_region failed");
	}
	completions_table_reg = reg;

	/* Everything starts at 0: counts 0, peer_completion {ordered 0, error_at 0}.
	 * completions_table_host is the persistent contiguous mirror; this method
	 * publishes it once to initialise the device copy. */
	completions_table_host.assign(completions_table_words, 0);
	if (get_device_copy().copy_to_device(completions_table_host.data(), *completions_table_reg, 0,
					     completions_table_words * sizeof(uint64_t)) != 0) {
		throw std::runtime_error("gdaki_completion_state: initial copy_to_device failed");
	}

	peer_bits.assign((size_t)nContexts_in * (size_t)nranks_in, {});
	ordered_completed_count_per_peer.assign((size_t)nContexts_in * (size_t)nranks_in, 0);
	peer_error_at.assign((size_t)nContexts_in * (size_t)nranks_in, 0);
	has_error.assign((size_t)nContexts_in, 0);
	err_status.assign((size_t)nContexts_in, 0);
	err_qp_num.assign((size_t)nContexts_in, 0);
	err_peer.assign((size_t)nContexts_in, 0);
	err_pseq.assign((size_t)nContexts_in, 0);
	host_cq_.resize((size_t)nContexts_in);
}

struct fid_cq *gdaki_completion_state::open_cq(int ctx_id, struct fid_domain *domain, size_t cq_size,
					       struct fi_efa_ops_gda *gda_ops)
{
	host_cq_[(size_t)ctx_id] = std::make_unique<gdaki_host_cq>();
	host_cq_[(size_t)ctx_id]->open(domain, cq_size);
	host_cq_[(size_t)ctx_id]->build(gda_ops, ctx_id);
	return host_cq_[(size_t)ctx_id]->cq();
}

int gdaki_completion_state::progress(size_t max_iter)
{
	bool dirty = false;
	for (size_t i = 0; i < host_cq_.size(); ++i) {
		if (!host_cq_[i]) continue;
		if (progress_cq(*host_cq_[i], (int)i, max_iter) > 0) {
			dirty = true;
		}
	}
	if (dirty) {
		return publish();
	}
	return 0;
}

bool gdaki_completion_state::query_error(std::string &msg) const
{
	const int idx = error_ctx_plus_one.load(std::memory_order_acquire);
	if (idx == 0) return false;
	const int ctx_id = idx - 1;
	msg = "GDAKI CQ completion error on ctx" + std::to_string(ctx_id) + ": status=" +
		  std::to_string((unsigned)err_status[ctx_id]) + " (" +
		  nccl_ofi_gin_gdaki_comp_status_str(err_status[ctx_id]) +
		  ") qp_num=" + std::to_string((unsigned)err_qp_num[ctx_id]) +
		  " peer=" + std::to_string((unsigned)err_peer[ctx_id]) +
		  " pseq=" + std::to_string((unsigned)err_pseq[ctx_id]);
	return true;
}

uint32_t gdaki_completion_state::progress_cq(gdaki_host_cq &cq, int ctx_id, size_t max_iter)
{
	uint32_t consumed = 0;
	uint64_t *table_host = host();
	const size_t ctx_off = ctx_word(ctx_id);
	const size_t peer_off = peer_word_base_for(ctx_id);

	for (size_t iter = 0; iter < max_iter; ++iter) {
		const uint32_t index = (uint32_t)((cq.cc + consumed) & cq.cq_mask);
		const uint8_t *entry = cq.cq_buf + ((size_t)index * cq.entry_size);
		const struct efa_io_cdesc_common *cqe =
			reinterpret_cast<const struct efa_io_cdesc_common *>(entry);

		const int expect_phase =
			cq.phase ^ (int)((((cq.cc & cq.cq_mask) + consumed) >> cq.cq_size_shift));

		/* Acquire pairs with the NIC's DMA write: the phase bit is the flag and the
		 * rest of the CQE is the payload, so the payload must not be read before the
		 * flag is seen. A full host barrier is required. */
		const uint8_t flags = __atomic_load_n(&cqe->flags, __ATOMIC_ACQUIRE);
		if (GDAKI_CDESC_FLAG_PHASE(flags) != expect_phase) {
			break;   /* ring empty at this position */
		}

		const uint8_t status = cqe->status;
		const uint16_t req_id = cqe->req_id;
		const uint16_t qp_num = cqe->qp_num;

		/* Every CQE on this CQ is a local TX completion for one of its posters. This
		 * bumps the per-context drain count in the host mirror (published once at the
		 * end of the pass). Per-QP completion lives in the endpoint's NIC counter; the
		 * pass keeps qp_num only for error reporting. */
		table_host[ctx_off] += 1;

		/* req_id encodes attribution: peer in the high bits, pseq in the low
		 * NCCL_OFI_GDAKI_PSEQ_BITS. */
		const uint16_t peer = (uint16_t)(req_id >> NCCL_OFI_GDAKI_PSEQ_BITS);
		const uint32_t pseq = (uint32_t)(req_id & (NCCL_OFI_GDAKI_PEER_WINDOW - 1));

		if (OFI_UNLIKELY(peer >= (uint16_t)nranks)) {
			NCCL_OFI_WARN("GDAKI CQ progress: ctx%d req_id=0x%x peer=%u >= nranks=%d",
				      ctx_id, req_id, peer, nranks);
			++consumed;
			continue;
		}

		const size_t slot = peer_slot(ctx_id, peer);
		peer_bits[slot][pseq >> 6] |= (1ull << (pseq & 63));

		if (OFI_UNLIKELY(status != EFA_IO_COMP_STATUS_OK)) {
			NCCL_OFI_WARN("GDAKI CQ error on ctx%d: status=%u (%s) qp_num=%u peer=%u pseq=%u",
				      ctx_id, status,
				      nccl_ofi_gin_gdaki_comp_status_str(status), qp_num, peer, pseq);
			/* error_at is biased +1 (0 means none); keep the earliest failing pseq. */
			const uint32_t biased = pseq + 1;
			if (peer_error_at[slot] == 0 || biased < peer_error_at[slot]) {
				peer_error_at[slot] = biased;
			}
			if (!has_error[ctx_id]) {
				has_error[ctx_id] = 1;
				err_status[ctx_id] = status;
				err_qp_num[ctx_id] = qp_num;
				err_peer[ctx_id] = peer;
				err_pseq[ctx_id] = pseq;
				int none = 0;
				error_ctx_plus_one.compare_exchange_strong(none, ctx_id + 1,
						std::memory_order_release, std::memory_order_relaxed);
			}
		}

		/* Advance the peer's ordered count inline: extend the contiguous run from its
		 * current value, clearing the bits it consumes. An out-of-order arrival only
		 * sets its bit above; a gap-filling completion extends the run. countr_one
		 * takes the whole run of set bits at the current position in one step, so the
		 * scan costs one iteration per 64-bit word instead of one per completed write.
		 * Then write this peer's {ordered, error_at} into the mirror. */
		auto &bits = peer_bits[slot];
		uint32_t &upto = ordered_completed_count_per_peer[slot];
		for (;;) {
			const uint32_t pos = upto & (NCCL_OFI_GDAKI_PEER_WINDOW - 1u);
			const uint32_t off = pos & 63u;
			/* Shifting the word down to the current position discards the bits
			 * already consumed and feeds in zeros above, so the run this counts
			 * cannot reach past the word. */
			const uint32_t run =
				(uint32_t)std::countr_one(bits[pos >> 6] >> off);
			if (run == 0) {
				break;
			}
			/* Clear the run just consumed. A whole word of set bits needs the
			 * all-ones mask, which (1ull << 64) cannot express. */
			bits[pos >> 6] &=
				~((run == 64u) ? ~0ull : (((1ull << run) - 1ull) << off));
			upto += run;
			/* A run that stopped before the word boundary ended on a bit that is
			 * not set, so the contiguous prefix ends there. */
			if (off + run < 64u) {
				break;
			}
		}
		auto *pc = reinterpret_cast<struct nccl_ofi_gin_gdaki_peer_completion *>(
			&table_host[peer_off + (size_t)peer]);
		pc->ordered_completed_count_per_peer = upto;
		pc->error_at = peer_error_at[slot];
		++consumed;
	}

	if (consumed > 0) {
		cq.phase = cq.phase ^ (int)((((cq.cc & cq.cq_mask) + consumed) >> cq.cq_size_shift));
		cq.cc += consumed;
	}
	return consumed;
}

int gdaki_completion_state::publish()
{
	return get_device_copy().copy_to_device(completions_table_host.data(), *completions_table_reg, 0,
						completions_table_words * sizeof(uint64_t));
}
