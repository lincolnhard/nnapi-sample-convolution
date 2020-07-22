#include "simple_model.hpp"
#include <random>
#include <cmath>

int main (int argc, char *argv[])
{
    std::random_device rd;
    std::mt19937 gen = std::mt19937(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    auto randfunc = std::bind(dis, gen);


    // NNAPI default data layout NHWC
    const uint32_t NET_W = 224;
    const uint32_t NET_H = 224;
    const uint32_t NET_C = 3;
    const uint32_t NET_IN_SIZE = NET_W * NET_H * NET_C;

    // Input buffers
    float *indataptr = new float[NET_IN_SIZE];
    for (int i = 0; i < NET_IN_SIZE; ++i)
    {
        indataptr[i] = randfunc();
    }

    // Convolution weight, bias
    const uint32_t KERNEL_N = 32;
    const uint32_t KERNEL_H = 3;
    const uint32_t KERNEL_W = 3;
    const uint32_t KERNEL_C = 3;
    const int32_t PAD = 1;
    const int32_t STRIDE = 1;
    const uint32_t KERNEL_SIZE = KERNEL_N * KERNEL_H * KERNEL_W * KERNEL_C;
    float *weightptr = new float[KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i)
    {
        weightptr[i] = randfunc();
    }

    float *biasptr = new float[KERNEL_N];
    for (int i = 0; i < KERNEL_N; ++i)
    {
        biasptr[i] = randfunc();
    }

    // Start to build
    SimpleModel nnModel;
    nnModel.addTensor("data", {1, NET_H, NET_W, NET_C});
    nnModel.addTensor("conv1_weight", {KERNEL_N, KERNEL_H, KERNEL_W, KERNEL_C}, weightptr);
    nnModel.addTensor("conv1_bias", {KERNEL_N}, biasptr);
    nnModel.conv2d("conv1", "data", "conv1_weight", "conv1_bias",
                    PAD, PAD, PAD, PAD, STRIDE, STRIDE, ANEURALNETWORKS_FUSED_NONE, "conv1_out");
    
    // Set input, output
    nnModel.setInputOps("data", indataptr);
    nnModel.setOutputOps("conv1_out");
    
    // compile
    nnModel.compile();

    // execute
    nnModel.execute();

    // validatation
    float *goldenRef = new float[NET_W * NET_H * KERNEL_N];
    for (uint32_t n = 0; n < KERNEL_N; ++n)
    {
        for (uint32_t y = 0; y < NET_H; ++y)
        {
            for (uint32_t x = 0; x < NET_W; ++x)
            {
                float sum = 0.0f;
                for (uint32_t j = 0; j < KERNEL_H; ++j)
                {
                    for (uint32_t i = 0; i < KERNEL_W; ++i)
                    {
                        for (uint32_t k = 0; k < KERNEL_C; ++k)
                        {
                            if (((x - PAD + i) < 0) || ((y - PAD + j) < 0) || ((x - PAD + i) >= NET_W) || ((y - PAD + j) >= NET_H))
                            {
                                continue;
                            }

                            float w = weightptr[k + i * (KERNEL_C) + j * (KERNEL_C * KERNEL_W) + n * (KERNEL_C * KERNEL_W * KERNEL_H)];
                            float p = indataptr[k + (x - PAD + i) * (KERNEL_C) + (y - PAD + j) * (KERNEL_C * NET_W)];
                            sum += (w * p);
                        }
                    }
                }
                // std::cout << "x: " << x << ", y: " << y << ", sum: " << sum << std::endl;
                goldenRef[n + x * KERNEL_N + y * (NET_W * KERNEL_C)] = sum + biasptr[n];
            }
        }
    }

    std::vector<float *> results = nnModel.getOutput();


    float diffsum = 0;
    for (int i = 0; i < NET_C * NET_W; ++i)
    {
        diffsum += std::fabsf(goldenRef[i] - results[0][i]);
    }
    std::cout << diffsum << std::endl;


    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;
    delete [] goldenRef;

    return 0;
}