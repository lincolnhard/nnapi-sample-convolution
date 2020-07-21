#include "simple_model.hpp"
#include <random>


int main (int argc, char *argv[])
{
    std::random_device rd;
    std::mt19937 gen = std::mt19937(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    auto randfunc = std::bind(dis, gen);


    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 224;
    const uint32_t NET_HEIGHT = 224;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // Input buffers
    float *indataptr = new float[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, randfunc());

    // Convolution weight, bias
    const uint32_t KERNEL_N = 32;
    const uint32_t KERNEL_H = 3;
    const uint32_t KERNEL_W = 3;
    const uint32_t KERNEL_C = 3;

    const uint32_t KERNEL_SIZE = KERNEL_N * KERNEL_H * KERNEL_W * KERNEL_C;
    float *weightptr = new float[KERNEL_SIZE];
    std::fill(weightptr, weightptr + KERNEL_SIZE, randfunc());
    float *biasptr = new float[KERNEL_N];
    std::fill(biasptr, biasptr + KERNEL_N, randfunc());

    // Start to build
    SimpleModel nnModel;
    nnModel.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS});
    nnModel.addTensor("conv1_weight", {KERNEL_N, KERNEL_H, KERNEL_W, KERNEL_C}, weightptr);
    nnModel.addTensor("conv1_bias", {KERNEL_N}, biasptr);
    nnModel.conv2d("conv1", "data", "conv1_weight", "conv1_bias",
                    1, 1, 1, 1, 1, 1, ANEURALNETWORKS_FUSED_RELU6, "conv1_out");
    
    // Set input, output
    nnModel.setInputOps("data", indataptr);
    nnModel.setOutputOps("conv1_out");
    
    // compile
    nnModel.compile();

    // execute
    nnModel.execute();

    return 0;
}