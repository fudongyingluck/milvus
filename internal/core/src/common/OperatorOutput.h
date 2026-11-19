// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <memory>

#include "common/EasyAssert.h"

namespace milvus {

enum class OperatorOutputKind {
    Rows,
    Bitmap,
};

// Common transport interface only. Storage and operations belong to each
// concrete result type. A null pointer means that no output is available.
class OperatorOutput {
 public:
    virtual ~OperatorOutput() = default;

    virtual size_t
    size() const noexcept = 0;

    virtual OperatorOutputKind
    Kind() const noexcept = 0;
};

using OperatorOutputPtr = std::shared_ptr<OperatorOutput>;

// Check at a consumer boundary, then use the concrete result type.
template <typename T>
std::shared_ptr<T>
OutputCast(const OperatorOutputPtr& output) {
    auto result = std::dynamic_pointer_cast<T>(output);
    AssertInfo(result != nullptr, "Unexpected operator output type");
    return result;
}

}  // namespace milvus
