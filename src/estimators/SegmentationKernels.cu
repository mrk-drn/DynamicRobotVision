
#include "estimators/SegmentationKernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SegmentationKernels {

namespace {

constexpr int INPUT_WIDTH = 768;
constexpr int INPUT_HEIGHT = 768;

constexpr int MASK_WIDTH = 192;
constexpr int MASK_HEIGHT = 192;
constexpr int MASK_SIZE = MASK_WIDTH * MASK_HEIGHT;

constexpr int NUM_MASK_COEFFS = 32;

constexpr int THREADS = 256;

void checkCuda(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            cudaGetErrorString(error)
        );
    }
}

template<typename T>
void reserveDevice(
    T*& pointer,
    std::size_t& capacity,
    std::size_t required
) {
    if (required <= capacity) {
        return;
    }

    T* replacement = nullptr;

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&replacement),
            required * sizeof(T)
        ),
        "cudaMalloc"
    );

    if (pointer != nullptr) {
        cudaFree(pointer);
    }

    pointer = replacement;
    capacity = required;
}

// Batched mask generation.

__global__ void generateMasksKernel(
    const float* coefficients,
    const float* prototypes,
    float* masks,
    int maskSize,
    int instanceCount
) {
    const int pixel =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int instance = blockIdx.y;

    if (
        pixel >= maskSize ||
        instance >= instanceCount
    ) {
        return;
    }

    const float* coeff =
        coefficients +
        static_cast<std::size_t>(instance) *
        NUM_MASK_COEFFS;

    float sum = 0.0f;

    #pragma unroll
    for (int channel = 0;
         channel < NUM_MASK_COEFFS;
         ++channel) {

        sum = fmaf(
            coeff[channel],
            prototypes[
                static_cast<std::size_t>(channel) *
                maskSize + pixel
            ],
            sum
        );
    }

    const float probability =
        sum >= 0.0f
        ? 1.0f / (1.0f + expf(-sum))
        : expf(sum) / (1.0f + expf(sum));

    masks[
        static_cast<std::size_t>(instance) *
        maskSize + pixel
    ] = probability;
}

// Bilinear sampling from one instance mask.

__device__ __forceinline__ float sampleMask(
    const float* mask,
    float x,
    float y
) {
    x = fminf(
        fmaxf(x, 0.0f),
        static_cast<float>(MASK_WIDTH - 1)
    );

    y = fminf(
        fmaxf(y, 0.0f),
        static_cast<float>(MASK_HEIGHT - 1)
    );

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));

    const int x1 = min(x0 + 1, MASK_WIDTH - 1);
    const int y1 = min(y0 + 1, MASK_HEIGHT - 1);

    const float dx = x - x0;
    const float dy = y - y0;

    const float a =
        mask[y0 * MASK_WIDTH + x0];

    const float b =
        mask[y0 * MASK_WIDTH + x1];

    const float c =
        mask[y1 * MASK_WIDTH + x0];

    const float d =
        mask[y1 * MASK_WIDTH + x1];

    const float top =
        fmaf(dx, b - a, a);

    const float bottom =
        fmaf(dx, d - c, c);

    return fmaf(
        dy,
        bottom - top,
        top
    );
}

// One thread per output pixel.

__global__ void composeMasksKernel(
    const unsigned char* frame,
    std::size_t frameStep,

    const float* masks,
    const int* boxes,
    const float* colors,

    unsigned char* visualization,
    std::size_t visualizationStep,

    unsigned char* combinedMask,
    std::size_t combinedMaskStep,

    unsigned char* colorMap,
    std::size_t colorMapStep,

    int width,
    int height,

    int resizedWidth,
    int resizedHeight,

    int padLeft,
    int padTop,

    float threshold,
    float opacity,

    int instanceCount
) {
    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* source =
        frame + y * frameStep + x * 3;

    unsigned char* destination =
        visualization +
        y * visualizationStep + x * 3;

    unsigned char* outputMask =
        combinedMask + y * combinedMaskStep + x;

    unsigned char* outputColor =
        colorMap + y * colorMapStep + x * 3;

    const float inputX =
        (static_cast<float>(x) + 0.5f) *
        resizedWidth / width +
        padLeft;

    const float inputY =
        (static_cast<float>(y) + 0.5f) *
        resizedHeight / height +
        padTop;

    const float maskX =
        inputX *
        (static_cast<float>(MASK_WIDTH) / INPUT_WIDTH)
        - 0.5f;

    const float maskY =
        inputY *
        (static_cast<float>(MASK_HEIGHT) / INPUT_HEIGHT)
        - 0.5f;

    float colorB = 0.0f;
    float colorG = 0.0f;
    float colorR = 0.0f;

    bool foreground = false;

    // Preserve existing composition behavior:
    // later overlapping instances overwrite earlier ones.

    for (int i = 0; i < instanceCount; ++i) {

        const int* box = boxes + i * 4;

        const int left = box[0];
        const int top = box[1];
        const int boxWidth = box[2];
        const int boxHeight = box[3];

        if (
            x < left ||
            y < top ||
            x >= left + boxWidth ||
            y >= top + boxHeight
        ) {
            continue;
        }

        const float* instanceMask =
            masks +
            static_cast<std::size_t>(i) *
            MASK_SIZE;

        const float probability =
            sampleMask(
                instanceMask,
                maskX,
                maskY
            );

        if (probability <= threshold) {
            continue;
        }

        const float* color = colors + i * 3;

        colorB = color[0];
        colorG = color[1];
        colorR = color[2];

        foreground = true;
    }

    if (foreground) {

        // Original blended visualization.

        destination[0] = static_cast<unsigned char>(
            fminf(
                fmaxf(
                    (1.0f - opacity) * source[0] +
                    opacity * colorB,
                    0.0f
                ),
                255.0f
            )
        );

        destination[1] = static_cast<unsigned char>(
            fminf(
                fmaxf(
                    (1.0f - opacity) * source[1] +
                    opacity * colorG,
                    0.0f
                ),
                255.0f
            )
        );

        destination[2] = static_cast<unsigned char>(
            fminf(
                fmaxf(
                    (1.0f - opacity) * source[2] +
                    opacity * colorR,
                    0.0f
                ),
                255.0f
            )
        );

        // Binary mask.

        *outputMask = 255;

        // Pure segmentation color.

        outputColor[0] =
            static_cast<unsigned char>(colorB);

        outputColor[1] =
            static_cast<unsigned char>(colorG);

        outputColor[2] =
            static_cast<unsigned char>(colorR);
    }
    else {

        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];

        *outputMask = 0;

        outputColor[0] = 0;
        outputColor[1] = 0;
        outputColor[2] = 0;
    }
}

} // namespace

struct Processor::Impl {

    float* dPrototypes = nullptr;
    float* dCoefficients = nullptr;
    float* dMasks = nullptr;

    int* dBoxes = nullptr;
    float* dColors = nullptr;

    unsigned char* dFrame = nullptr;
    unsigned char* dVisualization = nullptr;
    unsigned char* dCombinedMask = nullptr;
    unsigned char* dColorMap = nullptr;

    std::size_t prototypeCapacity = 0;
    std::size_t coefficientCapacity = 0;
    std::size_t maskCapacity = 0;

    std::size_t boxCapacity = 0;
    std::size_t colorCapacity = 0;

    std::size_t frameCapacity = 0;
    std::size_t visualizationCapacity = 0;
    std::size_t combinedMaskCapacity = 0;
    std::size_t colorMapCapacity = 0;

    cudaStream_t stream = nullptr;

    Impl() {
        checkCuda(
            cudaStreamCreateWithFlags(
                &stream,
                cudaStreamNonBlocking
            ),
            "cudaStreamCreateWithFlags"
        );
    }

    ~Impl() {

        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }

        cudaFree(dPrototypes);
        cudaFree(dCoefficients);
        cudaFree(dMasks);

        cudaFree(dBoxes);
        cudaFree(dColors);

        cudaFree(dFrame);
        cudaFree(dVisualization);
        cudaFree(dCombinedMask);
        cudaFree(dColorMap);

        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

Processor::Processor()
    : impl_(std::make_unique<Impl>())
{
}

Processor::~Processor() = default;

Processor::Processor(Processor&&) noexcept = default;

Processor& Processor::operator=(Processor&&) noexcept = default;

void Processor::process(
    const cv::Mat& frame,
    const float* prototypes,
    const cv::Mat& coefficients,
    const std::vector<cv::Vec4i>& boxes,
    const std::vector<cv::Vec3f>& colors,
    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop,
    float maskThreshold,
    float opacity,
    cv::Mat& visualization,
    cv::Mat& combinedMask,
    cv::Mat& colorMap
) {
    if (
        frame.empty() ||
        prototypes == nullptr ||
        coefficients.empty() ||
        coefficients.cols != NUM_MASK_COEFFS ||
        boxes.empty() ||
        boxes.size() != colors.size()
    ) {
        throw std::invalid_argument(
            "Invalid segmentation CUDA input."
        );
    }

    if (frame.type() != CV_8UC3) {
        throw std::invalid_argument(
            "CUDA segmentation expects CV_8UC3 input."
        );
    }

    Impl& gpu = *impl_;

    const int width = frame.cols;
    const int height = frame.rows;

    const int instanceCount =
        static_cast<int>(boxes.size());

    const std::size_t pixelCount =
        static_cast<std::size_t>(width) * height;

    const std::size_t frameBytes =
        pixelCount * 3;

    const std::size_t maskBytes =
        static_cast<std::size_t>(instanceCount) *
        MASK_SIZE * sizeof(float);

    const std::size_t coefficientBytes =
        static_cast<std::size_t>(instanceCount) *
        NUM_MASK_COEFFS * sizeof(float);

    const std::size_t boxBytes =
        static_cast<std::size_t>(instanceCount) *
        4 * sizeof(int);

    const std::size_t colorBytes =
        static_cast<std::size_t>(instanceCount) *
        3 * sizeof(float);

    reserveDevice(
        gpu.dPrototypes,
        gpu.prototypeCapacity,
        NUM_MASK_COEFFS * MASK_SIZE
    );

    reserveDevice(
        gpu.dCoefficients,
        gpu.coefficientCapacity,
        coefficientBytes / sizeof(float)
    );

    reserveDevice(
        gpu.dMasks,
        gpu.maskCapacity,
        maskBytes / sizeof(float)
    );

    reserveDevice(
        gpu.dBoxes,
        gpu.boxCapacity,
        boxBytes / sizeof(int)
    );

    reserveDevice(
        gpu.dColors,
        gpu.colorCapacity,
        colorBytes / sizeof(float)
    );

    reserveDevice(
        gpu.dFrame,
        gpu.frameCapacity,
        frameBytes
    );

    reserveDevice(
        gpu.dVisualization,
        gpu.visualizationCapacity,
        frameBytes
    );

    reserveDevice(
        gpu.dCombinedMask,
        gpu.combinedMaskCapacity,
        pixelCount
    );

    // Allocate segmentation color map on GPU.

    reserveDevice(
        gpu.dColorMap,
        gpu.colorMapCapacity,
        frameBytes
    );

    checkCuda(
        cudaMemcpyAsync(
            gpu.dPrototypes,
            prototypes,
            NUM_MASK_COEFFS * MASK_SIZE * sizeof(float),
            cudaMemcpyHostToDevice,
            gpu.stream
        ),
        "Upload prototypes"
    );

    checkCuda(
        cudaMemcpyAsync(
            gpu.dCoefficients,
            coefficients.ptr<float>(),
            coefficientBytes,
            cudaMemcpyHostToDevice,
            gpu.stream
        ),
        "Upload coefficients"
    );

    checkCuda(
        cudaMemcpyAsync(
            gpu.dBoxes,
            boxes.data(),
            boxBytes,
            cudaMemcpyHostToDevice,
            gpu.stream
        ),
        "Upload boxes"
    );

    checkCuda(
        cudaMemcpyAsync(
            gpu.dColors,
            colors.data(),
            colorBytes,
            cudaMemcpyHostToDevice,
            gpu.stream
        ),
        "Upload colors"
    );

    checkCuda(
        cudaMemcpy2DAsync(
            gpu.dFrame,
            width * 3,
            frame.data,
            frame.step,
            width * 3,
            height,
            cudaMemcpyHostToDevice,
            gpu.stream
        ),
        "Upload frame"
    );

    // Launch 1: all instance masks.

    const dim3 maskGrid(
        (MASK_SIZE + THREADS - 1) / THREADS,
        instanceCount
    );

    generateMasksKernel<<<
        maskGrid,
        THREADS,
        0,
        gpu.stream
    >>>(
        gpu.dCoefficients,
        gpu.dPrototypes,
        gpu.dMasks,
        MASK_SIZE,
        instanceCount
    );

    checkCuda(
        cudaGetLastError(),
        "generateMasksKernel"
    );

    // Launch 2: composition, resampling and coloring.

    const dim3 block(16, 16);

    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    composeMasksKernel<<<
        grid,
        block,
        0,
        gpu.stream
    >>>(
        gpu.dFrame,
        width * 3,

        gpu.dMasks,
        gpu.dBoxes,
        gpu.dColors,

        gpu.dVisualization,
        width * 3,

        gpu.dCombinedMask,
        width,

        gpu.dColorMap,
        width * 3,

        width,
        height,

        resizedWidth,
        resizedHeight,

        padLeft,
        padTop,

        maskThreshold,
        opacity,

        instanceCount
    );

    checkCuda(
        cudaGetLastError(),
        "composeMasksKernel"
    );

    // Create output matrices.

    visualization.create(
        height,
        width,
        CV_8UC3
    );

    combinedMask.create(
        height,
        width,
        CV_8UC1
    );

    colorMap.create(
        height,
        width,
        CV_8UC3
    );

    // Download visualization.

    checkCuda(
        cudaMemcpy2DAsync(
            visualization.data,
            visualization.step,
            gpu.dVisualization,
            width * 3,
            width * 3,
            height,
            cudaMemcpyDeviceToHost,
            gpu.stream
        ),
        "Download visualization"
    );

    // Download combined binary mask.

    checkCuda(
        cudaMemcpy2DAsync(
            combinedMask.data,
            combinedMask.step,
            gpu.dCombinedMask,
            width,
            width,
            height,
            cudaMemcpyDeviceToHost,
            gpu.stream
        ),
        "Download combined mask"
    );

    // Download pure segmentation color map.

    checkCuda(
        cudaMemcpy2DAsync(
            colorMap.data,
            colorMap.step,
            gpu.dColorMap,
            width * 3,
            width * 3,
            height,
            cudaMemcpyDeviceToHost,
            gpu.stream
        ),
        "Download segmentation color map"
    );

    // All downloads and kernel operations complete here.

    checkCuda(
        cudaStreamSynchronize(gpu.stream),
        "Synchronize segmentation CUDA stream"
    );
}

} // namespace SegmentationKernels
