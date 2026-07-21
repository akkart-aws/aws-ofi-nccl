# 1.21.0~alpha3 (July 2026, pre-release)

Internal pre-release of the OFI NCCL Plugin 1.21 line for the EFA dev
installer 1.50.0. Same source as 1.21.0~alpha2 plus: the default GIN type
is now GDAKI, so installer builds engage the kernel-initiated data path
with no OFI_NCCL_GIN_TYPE runtime setting. Builds without --enable-gdaki
fall back to proxy mode by default (an explicit OFI_NCCL_GIN_TYPE=GDAKI
still fails init on such builds). Not a public release.

# 1.21.0~alpha2 (July 2026, pre-release)

Internal pre-release of the OFI NCCL Plugin 1.21 line for the EFA dev
installer 1.50.0 (Nvidia GIN GDAKI / counting-events testing). Same source
as 1.21.0~alpha1 (aws-ofi-nccl master @ cb33a4c) plus a packaging fix:
pass --enable-gdaki in the debian rules and RPM spec so the installer's
pipeline-built plugin binaries actually contain the GDAKI backend
(1.21.0~alpha1 binaries were compiled with GDAKI stubbed out). Not a
public release.

# 1.21.0~alpha1 (July 2026, pre-release)

Internal pre-release of the OFI NCCL Plugin 1.21 line for the EFA dev
installer 1.50.0 (Nvidia GIN GDAKI / counting-events testing). Built from
aws-ofi-nccl master @ cb33a4c with GDAKI enabled. Not a public release.

This file is a placeholder on the primary development branch of the
OFI NCCL Plugin so that "make dist" works properly.  Release branches
will have an accurate release history in this location, and each
release tarball will also have up to date release notes.

If you're looking for Plugin releases, please see the [Releases
Page](https://github.com/aws/aws-ofi-nccl/releases).
