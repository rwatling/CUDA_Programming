//From https://github.com/mnicely/nvml_examples

/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This is a header class that utilizes NVML library.
 */

#ifndef NVMLCLASS_H_
#define NVMLCLASS_H_ 1

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <nvml.h>

int constexpr size_of_vector { 100000 };
int constexpr nvml_device_name_buffer_size { 100 };

// *************** FOR ERROR CHECKING *******************
#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL
// *************** FOR ERROR CHECKING *******************

class nvmlClass {
  public:
    nvmlClass( int const &deviceID, std::string  &filename, std::string type ) :
        time_steps_ {}, filename_ { filename }, outfile_ {}, start_stop_file_ {},
        device_ {}, loop_ { false } , start_flag_ { false }, stop_flag_ {false}, type_ { type } {

        char name[nvml_device_name_buffer_size];

        // Initialize NVML library
        NVML_RT_CALL( nvmlInit( ) );

        // Query device handle
        NVML_RT_CALL( nvmlDeviceGetHandleByIndex( deviceID, &device_ ) );

        // Query device name
        NVML_RT_CALL( nvmlDeviceGetName( device_, name, nvml_device_name_buffer_size ) );

        // Reserve memory for data
        time_steps_.reserve( size_of_vector );

        // Open file
        outfile_.open( filename_, std::ios::out );

        // Print header
        printHeader( );
    }

    ~nvmlClass( ) {

        NVML_RT_CALL( nvmlShutdown( ) );
    }

    void getStats( ) {

        stats device_stats {};
        loop_ = true;
        start_flag_ = false;
        stop_flag_ = false;

        while ( loop_ ) {
            device_stats.timestamp = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
            NVML_RT_CALL( nvmlDeviceGetTemperature( device_, NVML_TEMPERATURE_GPU, &device_stats.temperature ) );
            NVML_RT_CALL( nvmlDeviceGetPowerUsage( device_, &device_stats.powerUsage ) );
            NVML_RT_CALL( nvmlDeviceGetEnforcedPowerLimit( device_, &device_stats.powerLimit ) );
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_GRAPHICS, &device_stats.graphicsClock));
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_MEM, &device_stats.memClock));
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_SM, &device_stats.smClock));

            time_steps_.push_back( device_stats );

            if (start_flag_) {
              start_stop_file_ << "type,timestep,power\n";
              start_stop_file_ << type_ << "," << time_steps_.size() << "," << device_stats.powerUsage <<"\n";
              start_flag_ = false;
            } else if (stop_flag_) {
              start_stop_file_ << type_ << "," <<  time_steps_.size() << "," << device_stats.powerUsage <<"\n";
              start_stop_file_.close();
              stop_flag_ = false;
            }

            std::this_thread::sleep_for( std::chrono::microseconds(500) );
        }

        writeData();
    }

    void killThread( ) {
        // Retrieve a few empty samples
        std::this_thread::sleep_for( std::chrono::seconds(1) );

        // Set loop to false to exit while loop
        loop_ = false;
    }

    void new_experiment(std::string  &filename, std::string &type) {
      filename_ = filename;
      type_ = type;

      // Open file
      outfile_.open( filename_, std::ios::out );

      //Start stop name
      start_stop_name_ = "./start_stop_";
      start_stop_name_.append(type);
      start_stop_name_.append(".csv");

      //Open start start stop
      start_stop_file_.open(start_stop_name_, std::ios::out);

      // Print header
      printHeader( );
    }

    void log_start() {
      //Start stop name
      start_stop_name_ = "./start_stop_";
      start_stop_name_.append(type_);
      start_stop_name_.append(".csv");

      //Open start start stop
      start_stop_file_.open(start_stop_name_, std::ios::out);

      // Retrieve a few empty samples
      std::this_thread::sleep_for( std::chrono::seconds(1) );

      start_flag_ = true;
    }

    void log_stop() {
      stop_flag_ = true;
    }

  private:
    typedef struct _stats {
        std::time_t        timestamp;
        uint               temperature;
        uint               powerUsage;
        uint               powerLimit;
        uint               graphicsClock;
        uint               smClock;
        uint               memClock;
    } stats;

    std::vector<std::string> names_ = { "timestep",
                                        "timestamp",
                                        "temperature_gpu",
                                        "type",
                                        "power_draw_mW",
                                        "power_limit_mW",
                                        "g_clock_freq_mhz",
                                        "mem_clock_freq_mhz",
                                        "sm_clock_freq_mhz"};

    std::vector<stats> time_steps_;
    std::string        filename_;
    std::string        start_stop_name_;
    std::string        type_;
    std::ofstream      outfile_;
    std::ofstream      start_stop_file_;
    nvmlDevice_t       device_;
    bool               loop_;
    bool               start_flag_;
    bool               stop_flag_;

    void printHeader( ) {

        // Print header
        for ( int i = 0; i < ( static_cast<int>( names_.size( ) ) - 1 ); i++ )
            outfile_ << names_[i] << ", ";
        // Leave off the last comma
        outfile_ << names_[static_cast<int>( names_.size( ) ) - 1];
        outfile_ << "\n";
    }

    void writeData( ) {

        //printf( "\nWriting NVIDIA-SMI data -> %s\n\n", filename_.c_str( ) );

        // Print data
        for ( int i = 0; i < static_cast<int>( time_steps_.size( ) ); i++ ) {
            outfile_ << i << ", "
                     << time_steps_[i].timestamp << ","
                     << time_steps_[i].temperature << ", "
                     << type_ << ", "
                     << time_steps_[i].powerUsage << ", "  // mW
                     << time_steps_[i].powerLimit << ","  // mW
                     << time_steps_[i].graphicsClock << "," //MHz
                     << time_steps_[i].memClock << "," //MHz
                     << time_steps_[i].smClock << "\n"; //MHz
        }
        outfile_.close( );
    }
};

#endif /* NVMLCLASS_H_ */
