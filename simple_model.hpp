#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <functional>

#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/system_properties.h>
#include <sys/mman.h>


#define CHECK_NNAPI_ERROR(status)                                                                   \
        if (status != ANEURALNETWORKS_NO_ERROR)                                                     \
        {                                                                                           \
            std::cerr << status << ", line: " <<__LINE__ << std::endl;                             \
            exit(1);                                                                                \
        }



class TensorType
{
public:
    uint32_t index;
    std::vector<uint32_t> dimensions;
    uint32_t sizeBytes;
    std::string name;
    float *data;
    int fd;                                     // for shared memory
    ANeuralNetworksMemory *nnMemPtr;            // mapped memory to shared memory
};

class SimpleModel
{
public:
    SimpleModel(void);
    ~SimpleModel(void);
    void addTensor (std::string name, std::vector<uint32_t> dims, const void *srcbuffer = nullptr);

    void conv2d (const std::string& name, const std::string& input, const std::string& weight,
                const std::string& bias, int32_t padLeft, int32_t padRight,
                int32_t padTop, int32_t padBottom, int32_t strideX, int32_t strideY,
                FuseCode fusecode, const std::string& output);

    void setInputOps (std::string name, float *dataptr);
    void setOutputOps (std::string name);
    void compile(void);
    void execute(void);
    std::vector<float *> getOutput(void);

private:
    uint32_t opIdx;
    std::map<std::string, uint32_t> operandIdxes;
    std::map<std::string, std::vector<uint32_t>> shapeIdxes;
    std::vector<TensorType> inputTensors;
    std::vector<TensorType> outputTensors;

    ANeuralNetworksModel *model;
    ANeuralNetworksCompilation *compilation;
    ANeuralNetworksExecution *execution;
    ANeuralNetworksEvent *event;
    std::vector<ANeuralNetworksDevice *> devices;

    size_t getElementSize(int32_t opType);
};

