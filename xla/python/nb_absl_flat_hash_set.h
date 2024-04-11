/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_NB_ABSL_FLAT_HASH_SET_H_
#define XLA_PYTHON_NB_ABSL_FLAT_HASH_SET_H_

#include "absl/container/flat_hash_set.h"
#include "nanobind/nanobind.h"  // from @nanobind
#include "nanobind/stl/detail/nb_set.h"  // from @nanobind

namespace nanobind {
namespace detail {

template <typename Key, typename Hash, typename Eq, typename Alloc>
struct type_caster<absl::flat_hash_set<Key, Hash, Eq, Alloc>>
    : set_caster<absl::flat_hash_set<Key, Hash, Eq, Alloc>, Key> {};

}  // namespace detail
}  // namespace nanobind

#endif  // XLA_PYTHON_NB_ABSL_FLAT_HASH_SET_H_
