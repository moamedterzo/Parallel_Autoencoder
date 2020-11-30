################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Parallel_autoencoder.cpp \
../src/node_accumulator_autoencoder.cpp \
../src/node_autoencoder.cpp \
../src/node_cell_autoencoder.cpp \
../src/node_master_autoencoder.cpp \
../src/node_single_autoencoder.cpp \
../src/samples_manager.cpp 

OBJS += \
./src/Parallel_autoencoder.o \
./src/node_accumulator_autoencoder.o \
./src/node_autoencoder.o \
./src/node_cell_autoencoder.o \
./src/node_master_autoencoder.o \
./src/node_single_autoencoder.o \
./src/samples_manager.o 

CPP_DEPS += \
./src/Parallel_autoencoder.d \
./src/node_accumulator_autoencoder.d \
./src/node_autoencoder.d \
./src/node_cell_autoencoder.d \
./src/node_master_autoencoder.d \
./src/node_single_autoencoder.d \
./src/samples_manager.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	/usr/bin/mpic++ -I"/home/giovanni/git/repository/Parallel_autoencoder/opencv/include" -I../opencv_release/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


