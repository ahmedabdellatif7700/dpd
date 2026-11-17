# syntax=docker/dockerfile:1
#
# The dpd stage's image for the Jetson Orin Nano (JetPack 7, L4T R39.2.1, CUDA 13.2 driver):
# C++ with hand-written CUDA kernels (cpp/). Follows the user's Jetson CUDA template
# (jetson-llama-cuda: the CUDA 13.2 devel base, the compat workaround, CMake + Ninja for sm_87,
# BuildKit caches for the build and ccache). Built from the repository root:
#   make -C docker build          (docker compose -f docker/compose.yaml build)
#
# One addition to the template: the binaries are copied into the CUDA runtime image, so the
# stage runs without the compiler and the toolkit (the devel image is 10.4 GB).

ARG CUDA_VERSION=13.2.0

# ---- build ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Disable CUDA forward-compatibility libraries on Jetson.
# The NVIDIA container runtime supplies the host CUDA driver.
# This avoids the CDI enable-cuda-compat crash in toolkit 1.19.1.
RUN for dir in \
        /usr/local/cuda-13.2/compat \
        /usr/local/cuda-13.2/compat_orin; do \
        if [ -d "$dir" ]; then \
            mv "$dir" "$dir.disabled"; \
        fi; \
    done

# Install build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        build-essential \
        ccache && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY cpp/ cpp/

# Build for SM 8.7 (the Orin).
ARG CUDA_ARCH=87
RUN --mount=type=cache,id=arof-dpd-build-132,target=/src/cpp/build,sharing=locked \
    --mount=type=cache,id=arof-dpd-ccache,target=/root/.cache/ccache,sharing=locked \
    cmake -S cpp -B cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache && \
    cmake --build cpp/build -j 2 && \
    mkdir -p /opt/arof-dpd && \
    cp -a cpp/bin/. /opt/arof-dpd/

# ---- run ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# The same compat workaround as the build image.
RUN for dir in \
        /usr/local/cuda-13.2/compat \
        /usr/local/cuda-13.2/compat_orin; do \
        if [ -d "$dir" ]; then \
            mv "$dir" "$dir.disabled"; \
        fi; \
    done

COPY --from=build /opt/arof-dpd/ /opt/arof-dpd/
ENV PATH=/opt/arof-dpd:${PATH}

# Step 3: the dpd stage becomes the command; gpu_check stays as the image's self-test.
CMD ["gpu_check"]
