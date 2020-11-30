################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Parallel_autoencoder.cpp \
../src/custom_utils.cpp \
../src/samples_manager.cpp 

OBJS += \
./src/Parallel_autoencoder.o \
./src/custom_utils.o \
./src/samples_manager.o 

CPP_DEPS += \
./src/Parallel_autoencoder.d \
./src/custom_utils.d \
./src/samples_manager.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	/usr/bin/mpic++ -I"/home/giovanni/git/repository/Parallel_autoencoder/opencv/include" -I../opencv_release/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


