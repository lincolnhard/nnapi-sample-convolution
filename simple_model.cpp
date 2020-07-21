#include "simple_model.hpp"

#include <numeric>
#include <algorithm>
#include <cassert>

#include <unistd.h>


SimpleModel::~SimpleModel()
{
    for (auto it: outputTensors)
    {
        munmap(it.data, it.sizeBytes);
        ANeuralNetworksMemory_free(it.nnMemPtr);
        close(it.fd);
    }
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
}

SimpleModel::SimpleModel()
{
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_create(&model) );
    opIdx = 0;
}


void SimpleModel::addTensor (std::string name,
                            std::vector<uint32_t> dims,
                            const void *srcbuffer
                            )
{
    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.data();
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name] = opIdx;
    shapeIdxes[name] = dims;

    if (srcbuffer != nullptr)
    {
        const size_t bytes = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()) * sizeof(float);
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, srcbuffer, bytes) );
    }
    ++opIdx;
}


void SimpleModel::conv2d (const std::string& name,
                        const std::string& input,
                        const std::string& weight,
                        const std::string& bias,
                        int32_t padLeft,
                        int32_t padRight,
                        int32_t padTop,
                        int32_t padBottom,
                        int32_t strideX,
                        int32_t strideY,
                        FuseCode fusecode,
                        const std::string& output
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    const auto weightIdx = operandIdxes.at(weight);
    const auto biasIdx = operandIdxes.at(bias);
    parameterIdxes.push_back(inputIdx);
    parameterIdxes.push_back(weightIdx);
    parameterIdxes.push_back(biasIdx);


    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType));
    operandIdxes[name + "_padLeft"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padLeft, sizeof(padLeft)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padRight"] = opIdx;
    CHECK_NNAPI_ERROR ( ANeuralNetworksModel_setOperandValue(model, opIdx, &padRight, sizeof(padRight)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padTop"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padTop, sizeof(padTop)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padBottom"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padBottom, sizeof(padBottom)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideX"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideX, sizeof(strideX)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideY"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideY, sizeof(strideY)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_activation"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &fusecode, sizeof(fusecode)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    const auto inDims = shapeIdxes.at(input);
    const auto wDims = shapeIdxes.at(weight);
    const uint32_t outN = inDims[0];
    const uint32_t outH = (inDims[1] - wDims[1] + padTop + padBottom) / strideY + 1;
    const uint32_t outW = (inDims[2] - wDims[2] + padLeft + padRight) / strideX + 1;
    uint32_t outC = wDims[0];

    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    operandType.dimensionCount = static_cast<uint32_t>(inDims.size());
    operandType.dimensions = outDims.data();
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );

}


void SimpleModel::setInputOps (std::string name, float* dataptr)
{
    uint32_t idx = operandIdxes.at(name);
    std::vector<uint32_t> shape = shapeIdxes.at(name);
    uint32_t sizebyte = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()) * sizeof(float);
    inputTensors.push_back({idx, shape, sizebyte, name, dataptr});
}


void SimpleModel::setOutputOps (std::string name)
{
    uint32_t idx = operandIdxes.at(name);
    std::vector<uint32_t> shape = shapeIdxes.at(name);
    uint32_t sizebyte = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()) * sizeof(float);
    int fd = ASharedMemory_create("an_optional_name", sizebyte);
    ANeuralNetworksMemory *memptr = nullptr;
    CHECK_NNAPI_ERROR( ANeuralNetworksMemory_createFromFd(sizebyte, PROT_READ | PROT_WRITE, fd, 0, &memptr) );
    float *dataptr = reinterpret_cast<float *>(mmap(nullptr, sizebyte, PROT_READ, MAP_SHARED, fd, 0) );
    outputTensors.push_back({idx, shape, sizebyte, name, dataptr, fd, memptr});
}


void SimpleModel::compile (void)
{
    std::vector<uint32_t> inputIndices;
    std::vector<uint32_t> outputIndices;
    for (auto it: inputTensors)
    {
        inputIndices.push_back(it.index);
    }
    for (auto it: outputTensors)
    {
        outputIndices.push_back(it.index);
    }
    // The values of constant and intermediate operands cannot be altered after the finish function is called
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_identifyInputsAndOutputs(model, inputIndices.size(), inputIndices.data(), outputIndices.size(), outputIndices.data()) );
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_finish(model) );
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_create(model, &compilation) );
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER) );
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_finish(compilation) );
}

void SimpleModel::execute (void)
{
    // Multiple concurrent execution instances could be created from the same compiled model.
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_create(compilation, &execution) );
    // Associate to the model inputs. Note that the index here uses the operand of the model input list, not all operand list
    for (size_t i = 0; i < inputTensors.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setInput(execution, static_cast<int32_t>(i), nullptr, inputTensors[i].data, inputTensors[i].sizeBytes) );
    }
    // Set the output tensor that will be filled by executing the model. Shared memory here to minimize the copies needed for getting the output data.
    // Note that the index here uses the operand of the model output list, not all operand list
    for (size_t i = 0; i < outputTensors.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setOutputFromMemory(execution, static_cast<int32_t>(i), nullptr, outputTensors[i].nnMemPtr, 0, outputTensors[i].sizeBytes) );
    }
    // Note that the execution here is asynchronous, event will be created to monitor the status of the execution.
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_startCompute(execution, &event) );
    // Wait until the completion of the execution. This could be done on a different thread.
    // By waiting immediately, we effectively make this a synchronous call.
    CHECK_NNAPI_ERROR( ANeuralNetworksEvent_wait(event) );

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
}

std::vector<float *> SimpleModel::getOutput(void)
{
    std::vector<float *> outputTensorPtrs;
    for (auto it: outputTensors)
    {
        outputTensorPtrs.push_back(it.data);
    }

    return outputTensorPtrs;
}
