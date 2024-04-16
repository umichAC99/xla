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
#include "xla/service/gpu/model/gpu_performance_model.h"
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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

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
  GpuCostModelStatsCollection cost_model_stats_1{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_2{
      TestGpuDeviceInfo::AMDMI210DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_3{
      TestGpuDeviceInfo::H100PCieDeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_4{
      TestGpuDeviceInfo::H100DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_5{
      TestGpuDeviceInfo::V100SXM216GBDeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_6{
      TestGpuDeviceInfo::A10080GBDeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  GpuCostModelStatsCollection cost_model_stats_7{
      TestGpuDeviceInfo::A10040GBDeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
  
  std::vector<std::string> device_names = {"a6000","amdmi210","h100_pcie","h100", "v100", "a100_80", "a100_40"};
    

    // GpuCostModelStatsCollectionTest(){
    //     cost_model_stats_.push_back(cost_model_stats_1);
    //     cost_model_stats_.push_back(cost_model_stats_2);
    //     cost_model_stats_.push_back(cost_model_stats_3);
    //     cost_model_stats_.push_back(cost_model_stats_4);
    //     cost_model_stats_.push_back(cost_model_stats_5);
    //     cost_model_stats_.push_back(cost_model_stats_6);
    //     cost_model_stats_.push_back(cost_model_stats_7);
    // }

};

TEST_F(GpuCostModelStatsCollectionTest, FusinInEntryComputation) {
    std::string base_path = "/xla/xla/service/gpu/model/out.txt";
    std::ifstream file(base_path); // Open the file
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
        }
        std::string firstline;
        std::getline(file,firstline);
        int num_layers = std::stoi(firstline);
        // file.close();

    std::string layer_hlo_strings[num_layers];

    // std::ifstream file1(base_path);
    std::string line;
    std::getline(file,line);
    // std::getline(file1,line);
    std::stringstream buffer;
    int i = 0;
    while(std::getline(file,line)){
        if(line == "zkn"){
            layer_hlo_strings[i] = buffer.str();
            i = i + 1;
            buffer.str("");
        }
        else{
            buffer << line;
        }
    }
    // std::cout << layer_hlo_strings[0] << std::endl;
    file.close();

    base_path = "/xla/xla/service/gpu/model/layer_times.txt";
    std::ofstream output_file(base_path);
    
    output_file << device_names[0] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_1.Run(module.get()).value());
        int warp_size = cost_model_stats_1.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_1.device_info_, cost_model_stats_1.cost_analysis_.flop_count(), num_threads);
        // double duration_double = absl::ToDoubleMicroseconds(duration);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
        
    }
    output_file <<"zkn"<<std::endl;

    output_file << device_names[1] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_2.Run(module.get()).value());
        int warp_size = cost_model_stats_2.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_2.device_info_, cost_model_stats_2.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
    }

    output_file <<"zkn"<<std::endl;
    output_file << device_names[2] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_3.Run(module.get()).value());
        int warp_size = cost_model_stats_3.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_3.device_info_, cost_model_stats_3.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;    }

    output_file <<"zkn"<<std::endl;
    output_file << device_names[3] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_4.Run(module.get()).value());
        int warp_size = cost_model_stats_4.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_4.device_info_, cost_model_stats_4.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
    }

    output_file <<"zkn"<<std::endl;
    output_file << device_names[4] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_5.Run(module.get()).value());
        int warp_size = cost_model_stats_5.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_5.device_info_, cost_model_stats_5.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
    }

    output_file <<"zkn"<<std::endl;
    output_file << device_names[5] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_6.Run(module.get()).value());
        int warp_size = cost_model_stats_6.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_6.device_info_, cost_model_stats_6.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
    }

    output_file <<"zkn"<<std::endl;
    output_file << device_names[6] + ":"<<std::endl;
    for(int i = 0; i<num_layers; i++){
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(layer_hlo_strings[i]));
        EXPECT_FALSE(cost_model_stats_7.Run(module.get()).value());
        int warp_size = cost_model_stats_7.device_info_.threads_per_warp();
        int num_threads = GetNumThreads(warp_size, GpuPerformanceWithCollectiveModel::kLL128NumThreads / 4,
                                        GpuPerformanceWithCollectiveModel::kLL128NumThreads, 512);
        absl::Duration duration = GpuPerformanceModelBase::ComputeTime(cost_model_stats_7.device_info_, cost_model_stats_7.cost_analysis_.flop_count(), num_threads);
        std::string durationString = absl::FormatDuration(duration);
        output_file << durationString << std::endl;
    }
    output_file <<"zkn"<<std::endl;


        
    
  


 // HloInstruction* rs_start =
        //     FindInstruction(module.get(), "all-reduce-start.1");

        // HloInstruction* producer =
        //         module->entry_computation()->GetInstructionWithName("conv0");


        // HloInstruction* root = module->entry_computation()->root_instruction();
        //   TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
        //                           root->backend_config<GpuBackendConfig>());
        //   const FusionBackendConfig& backend_config =
        //       gpu_config.fusion_backend_config();


            //   int64_t num_channels =
            //   std::max(min_nchannels, GetNcclMaxNumChannels(CollectiveAlgo::RING));
        //   std::cout << "Time:"<< GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(*rs_start,&(cost_model_stats_.cost_analysis_),cost_model_stats_.device_info_) << std::endl;
        //   std::cout << "Time1:" << GpuPerformanceModel::EstimateRunTimeForInstruction(producer, &(cost_model_stats_.cost_analysis_), GpuPerformanceModelOptions::Default()).exec_time << std::endl;
        // std::cout << "Flop_Count: "<<cost_model_stats_[0].cost_analysis_.flop_count() << std::endl;
        // std::cout<< "Num_of_threads: "<<num_threads<<std::endl;

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

// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);

//     if (argc > 1 && std::string(argv[1]) == "--test_arg") {
//         // Use the next argument as the file name
//         if (argc > 2) {
//             std::string fileName = argv[2];
//             std::cout << "File Name: " << fileName << std::endl;
//             // Use the fileName in your test
//         } else {
//             std::cerr << "Error: Missing file name argument." << std::endl;
//             return 1;
//         }
//     }
//     return RUN_ALL_TESTS();
// }
