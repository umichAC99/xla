/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include <stdint.h>

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

int GetNumThreads(int warp_size, int min_num_threads, int max_num_threads,
                  int default_num_threads) {
  int threads_from_env = default_num_threads;
  const char* env = std::getenv("NCCL_NTHREADS");
  if (env != nullptr) {
    CHECK(absl::SimpleAtoi(env, &threads_from_env));
  }
  int num_threads = threads_from_env;
  if (num_threads > 0) {
    if (num_threads % warp_size != 0) {
      num_threads = max_num_threads;
    } else if (num_threads > max_num_threads) {
      num_threads = max_num_threads;
    } else if (num_threads < min_num_threads) {
      num_threads = min_num_threads;
    }
  } else {
    num_threads = default_num_threads;
  }
  return num_threads;
}

class GpuCostModelStatsCollectionTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  GpuCostModelStatsCollection cost_model_stats_{
      TestGpuDeviceInfo::AMDMI210DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
};

TEST_F(GpuCostModelStatsCollectionTest, FusinInEntryComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

region_20.995 {
  Arg_1.997 = f32[] parameter(1)
  Arg_0.996 = f32[] parameter(0)
  ROOT add.589 = f32[] add(Arg_0.996, Arg_1.997)
}

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[1024,2048,2048]{2,1,0} parameter(2)
  p3 = f32[2048,2048,2048]{2,1,0} parameter(3)
  all-reduce-start.1 = f32[1024,2048,2048]{2,1,0} all-reduce-start(p2), channel_id=8, replica_groups={{0}}, to_apply=region_20.995, backend_config="{\"is_sync\":false}"
  all-reduce-start.2 = f32[2048,2048,2048]{2,1,0} all-reduce-start(p3), channel_id=10, replica_groups={{0}}, to_apply=region_20.995, backend_config="{\"is_sync\":false}"

  all-reduce-done.1 = f32[1024,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.1)
  all-reduce-done.2 = f32[2048,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.2)
  conv0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb

  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(conv0, all-reduce-done.1, all-reduce-done.2)
}
)"));

  EXPECT_FALSE(cost_model_stats_.Run(module.get()).value());

  int warp_size = cost_model_stats_.device_info_.threads_per_warp();
  int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                  GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
  HloInstruction* rs_start =
      FindInstruction(module.get(), "all-reduce-start.1");


//   HloInstruction* root = module->entry_computation()->root_instruction();
//   TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
//                           root->backend_config<GpuBackendConfig>());
//   const FusionBackendConfig& backend_config =
//       gpu_config.fusion_backend_config();


    //   int64_t num_channels =
    //   std::max(min_nchannels, GetNcclMaxNumChannels(CollectiveAlgo::RING));
    

  std::cout << "Time:"<< GpuPerformanceModelBase::ComputeTime(cost_model_stats_.device_info_, cost_model_stats_.cost_analysis_.flop_count(), num_threads) << std::endl;
  std::cout << "Time:"<< GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(*rs_start,&(cost_model_stats_.cost_analysis_),cost_model_stats_.device_info_) << std::endl;

  std::cout << "Flop_Count: "<<cost_model_stats_.cost_analysis_.flop_count() << std::endl;
  std::cout<< "Num_of_threads: "<<num_threads<<std::endl;


//   EXPECT_TRUE(backend_config.has_reification_cost());
//   EXPECT_GT(backend_config.reification_cost().end_to_end_cycles(), 0);
}

// TEST_F(GpuCostModelStatsCollectionTest, FusinInWhileComputation) {
//   TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
//     HloModule test_module

//     cond {
//       p = f32[16384]{0} parameter(0)
//       ROOT %constant.2 = pred[] constant(true)
//     }

//     log {
//       p = f32[16384]{0} parameter(0)
//       ROOT l = f32[16384]{0} log(p)
//     }

//     loop {
//       %p0 = f32[16384] parameter(0)
//       ROOT %res = f32[16384]{0} fusion(p0), kind=kInput, calls=log
//     }

//     ENTRY main {
//       %p0 = f32[16384] parameter(0)
//       ROOT %while = f32[16384] while(%p0), body=%loop, condition=%cond
//     })"));

//   EXPECT_FALSE(cost_model_stats_.Run(module.get()).value());

//   HloInstruction* root = module->entry_computation()
//                              ->root_instruction()
//                              ->while_body()
//                              ->root_instruction();
//   TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
//                           root->backend_config<GpuBackendConfig>());
//   const FusionBackendConfig& backend_config =
//       gpu_config.fusion_backend_config();

//   EXPECT_TRUE(backend_config.has_reification_cost());
//   EXPECT_GT(backend_config.reification_cost().end_to_end_cycles(), 0);
// }

}  // namespace gpu
}  // namespace xla
