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
#include <optional>
#include <utility>

#include "common/EasyAssert.h"
#include "common/OperatorOutput.h"
#include "common/Types.h"

namespace milvus {

enum class BitmapValue {
    AllZero,
    AllOne,
};

// Owns a data bitmap and optional validity in the same coordinate space.
// Bit meaning (predicate result or exclusion mask) is defined by the caller.
class NullableBitmap : public OperatorOutput {
 public:
    // Move existing storage; validity, when present, must match data's size.
    explicit NullableBitmap(
        TargetBitmap data,
        std::optional<TargetBitmap> validity = std::nullopt)
        : data_(std::move(data)), validity_(std::move(validity)) {
        AssertInfo(!validity_ || validity_->size() == data_.size(),
                   "Validity bitmap must have the same size as data");
    }

    explicit NullableBitmap(
        size_t size,
        BitmapValue data_value,
        std::optional<BitmapValue> validity_value = std::nullopt)
        : data_(size, data_value == BitmapValue::AllOne) {
        if (validity_value) {
            validity_.emplace(size, *validity_value == BitmapValue::AllOne);
        }
    }

    size_t
    size() const noexcept override {
        return data_.size();
    }

    OperatorOutputKind
    Kind() const noexcept override {
        return OperatorOutputKind::Bitmap;
    }

    // Always non-null. Existing TargetBitmap const views still allow writes.
    const TargetBitmap*
    Data() const {
        return &data_;
    }

    // Returns nullptr when validity was not provided.
    const TargetBitmap*
    Validity() const {
        return validity_ ? &*validity_ : nullptr;
    }

 private:
    TargetBitmap data_;
    std::optional<TargetBitmap> validity_;
};

using NullableBitmapPtr = std::shared_ptr<NullableBitmap>;

}  // namespace milvus
