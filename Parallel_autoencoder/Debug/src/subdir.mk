################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Parallel_autoencoder.cpp \
../src/autoencoder.cpp \
../src/custom_utils.cpp \
../src/rbm.cpp \
../src/samples_manager.cpp 

OBJS += \
./src/Parallel_autoencoder.o \
./src/autoencoder.o \
./src/custom_utils.o \
./src/rbm.o \
./src/samples_manager.o 

CPP_DEPS += \
./src/Parallel_autoencoder.d \
./src/autoencoder.d \
./src/custom_utils.d \
./src/rbm.d \
./src/samples_manager.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I"/home/giovanni/git/repository/Parallel_autoencoder/opencv/include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


