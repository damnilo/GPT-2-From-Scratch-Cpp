#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/CustomGemm.h"
#include "../NumCPP/include/Tensor.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-4f;
constexpr size_t kCudaCutoff = 65536;

enum class Path {
    Cpu,
    Cublas,
    Custom,
};

struct Case {
    const char* name;
    bool batched;
    size_t batch;
    size_t rows;
    size_t common;
    size_t cols;
};

void fill(Tensor& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = std::sin(static_cast<float>(i + 1)) * 0.1f;
    }
}

Tensor makeLeft(const Case& item) {
    Tensor tensor = item.batched
        ? Tensor({item.batch, item.rows, item.common}, 0.0f)
        : Tensor({item.rows, item.common}, 0.0f);
    fill(tensor);
    return tensor;
}

Tensor makeRight(const Case& item) {
    Tensor tensor = item.batched
        ? Tensor({item.batch, item.common, item.cols}, 0.0f)
        : Tensor({item.common, item.cols}, 0.0f);
    fill(tensor);
    return tensor;
}

void select(Path path) {
    Tensor::setAllowCudaGemm(path == Path::Cublas);
    Tensor::setAllowCustomGemm(path == Path::Custom);
}

Tensor multiply(Path path, const Case& item, const Tensor& left, const Tensor& right) {
    select(path);
    Tensor::resetGemmReport();
    return item.batched ? left.batchMatmul(right) : left.matmul(right);
}

bool close(const Tensor& expected, const Tensor& actual) {
    if (actual.getShape() != expected.getShape()) {
        return false;
    }

    const std::vector<float>& expected_data = expected.getData();
    const std::vector<float>& actual_data = actual.getData();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        const float diff = std::fabs(actual_data[i] - expected_data[i]);
        const float limit = kAtol + kRtol * std::fabs(expected_data[i]);
        if (diff > limit) {
            return false;
        }
    }
    return true;
}

void report(const std::string& name, bool ok, int& failures) {
    std::cout << name << (ok ? " correct" : " failed") << std::endl;
    if (!ok) {
        ++failures;
    }
}

size_t product(const Case& item) {
    return item.batch * item.rows * item.common * item.cols;
}

Tensor directCublas(const Case& item, const Tensor& left, const Tensor& right) {
    Tensor result = item.batched
        ? Tensor({item.batch, item.rows, item.cols}, 0.0f)
        : Tensor({item.rows, item.cols}, 0.0f);
    const bool ok = item.batched
        ? cudaBatchGemm(left.getData().data(), right.getData().data(), &result[0],
            item.batch, item.rows, item.common, item.cols)
        : cudaGemm(left.getData().data(), right.getData().data(), &result[0],
            item.rows, item.common, item.cols);
    if (!ok) {
        result = Tensor();
    }
    return result;
}

Tensor directCustom(const Case& item, const Tensor& left, const Tensor& right) {
    Tensor result = item.batched
        ? Tensor({item.batch, item.rows, item.cols}, 0.0f)
        : Tensor({item.rows, item.cols}, 0.0f);
    const bool ok = item.batched
        ? customBatchGemm(left.getData().data(), right.getData().data(), &result[0],
            item.batch, item.rows, item.common, item.cols)
        : customGemm(left.getData().data(), right.getData().data(), &result[0],
            item.rows, item.common, item.cols);
    if (!ok) {
        result = Tensor();
    }
    return result;
}

}

int main() {
    const Case cases[] = {
        {"small matmul", false, 1, 8, 5, 7},
        {"large matmul", false, 1, 64, 128, 64},
        {"batch matmul", true, 2, 64, 32, 64},
    };

    int failures = 0;
    const bool gpu = cudaGemmAvailable();
    if (!gpu) {
        std::cout << "GPU is not visible. cuBLAS and custom calls are skipped." << std::endl;
    }

    for (const Case& item : cases) {
        const Tensor left = makeLeft(item);
        const Tensor right = makeRight(item);
        const Tensor cpu = multiply(Path::Cpu, item, left, right);
        const GemmReport cpu_report = Tensor::gemmReport();
        report(std::string("cpu ") + item.name,
            !cpu_report.cublas && !cpu_report.custom && !cpu.empty(),
            failures);

        if (!gpu) {
            continue;
        }

        const bool above_cutoff = product(item) >= kCudaCutoff;
        const Tensor cublas = multiply(Path::Cublas, item, left, right);
        const GemmReport cublas_report = Tensor::gemmReport();
        if (above_cutoff) {
            report(std::string("cublas ") + item.name,
                cublas_report.cublas && !cublas_report.custom && close(cpu, cublas),
                failures);
        } else {
            report(std::string("cublas ") + item.name + " stays on cpu below cutoff",
                !cublas_report.cublas && !cublas_report.custom && close(cpu, cublas),
                failures);
            const Tensor direct = directCublas(item, left, right);
            report(std::string("cublas direct ") + item.name,
                !direct.empty() && close(cpu, direct),
                failures);
        }

        const Tensor custom = multiply(Path::Custom, item, left, right);
        const GemmReport custom_report = Tensor::gemmReport();
        report(std::string("custom ") + item.name,
            custom_report.custom && !custom_report.cublas && close(cpu, custom),
            failures);

        const Tensor custom_direct = directCustom(item, left, right);
        report(std::string("custom direct ") + item.name,
            !custom_direct.empty() && close(cpu, custom_direct),
            failures);
    }

    select(Path::Cublas);
    return failures == 0 ? 0 : 1;
}
