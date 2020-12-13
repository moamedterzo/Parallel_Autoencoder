/*
 * generators.h
 *
 *  Created on: 11 dic 2020
 *      Author: giovanni
 */

#ifndef GENERATORS_H_
#define GENERATORS_H_


#include <iostream>
#include <random>

namespace parallel_autoencoder
{
	//I seed sono stati presi dal seguente link: https://www.tutorialspoint.com/cplusplus-program-to-implement-the-linear-congruential-generator-for-pseudo-random-number-generation
	static std::default_random_engine generator_for_weights{ 1915290694 };
	static std::default_random_engine generator_for_hidden_sampling{ 1005338679 };
	static std::default_random_engine generator_for_visible_rec_noise { 629284700 };
	static std::default_random_engine generator_for_hidden_rec_sampling{ 741596485 };


	inline void skip_gen(std::default_random_engine& gen, const uint n)
	{
		//generate n pseudo-random numbers
		for(uint i = 0; i != n; i++)
			gen();
	}




	//Implement sampling for the sigmoid function
	//It's used an efficient version for sampling
	inline float sample_sigmoid_function(const float sigmoid_argument, std::default_random_engine& generator){

		//distribuzione uniforme tra 0 e 1
		static std::uniform_real_distribution<float> uniform_dis(0.0, 1.0);

		auto uniform_v = uniform_dis(generator);
		auto logit_v = logf(uniform_v / ( 1 - uniform_v));

		return sigmoid_argument > logit_v ? 1.0 : 0.0;
	}


	static std::normal_distribution<float> normal_dist(0, 0);
	inline void skip_gaussian_gen(std::default_random_engine& gen, const uint n)
	{
		//generate n pseudo-random numbers
		for(uint i = 0; i != n; i++)
			normal_dist(gen);
	}

	inline float sample_gaussian_noise(const float mean, const float variance,  std::default_random_engine& generator)
	{
		//std::normal_distribution<float> dist(mean, variance);

		//set parameters
		normal_dist.param(std::normal_distribution<float>::param_type(mean, variance));

		//auto gen = generator;
		auto value = normal_dist(generator);

		//static int a = 0;

		/*if (a++ < 4 && mean != 0)
		{
			std::string aaaaa = "";
			for(int i = 0; i < 8; i++)
				aaaaa += std::to_string(gen()) + "-";
			aaaaa += "\n";

			for(int i = 0; i < 8; i++)
				aaaaa += std::to_string(generator()) + "-";
			aaaaa += "\n";

			std::cout << aaaaa;
		}*/


		return value;
	}

	inline float sample_gaussian_noise(const float mean, std::default_random_engine& generator){

		return sample_gaussian_noise(mean, 1.0, generator);
	}




	inline void initialize_weight_matrix(matrix<float>& weights,
			const float rbm_initial_weights_mean, const float rbm_initial_weights_variance)
	{
		for(uint r = 0; r != weights.get_rows(); r++)
			for(uint c = 0; c != weights.get_cols(); c++)
				weights.at(r, c) = sample_gaussian_noise(rbm_initial_weights_mean, rbm_initial_weights_variance, generator_for_weights);
	}

	inline void initialize_weight_matrix(matrix<float>& weights,
			const float rbm_initial_weights_mean, const float rbm_initial_weights_variance,
			const uint n_visible_units, const uint n_hidden_units,
			const uint r_index, const uint c_index)
	{
		//si saltano le righe che non fanno parte della sottomatrice
		skip_gaussian_gen(generator_for_weights, r_index * n_hidden_units);

		for(uint r = 0; r != weights.get_rows(); r++)
		{
			//si saltano le colonne
			skip_gaussian_gen(generator_for_weights, c_index);

			for(uint c = 0; c != weights.get_cols(); c++)
				weights.at(r, c) = sample_gaussian_noise(rbm_initial_weights_mean, rbm_initial_weights_variance, generator_for_weights);

			//si saltano le colonne
			skip_gaussian_gen(generator_for_weights, n_hidden_units - c_index - weights.get_cols());
		}

		//si saltano le righe che non fanno parte della sottomatrice
		skip_gaussian_gen(generator_for_weights, (n_visible_units - r_index - weights.get_rows()) * n_hidden_units);
	}


	inline void sample_hidden_units(my_vector<float>& hidden_units, const my_vector<float>& hidden_biases)
	{
		for(uint i = 0; i != hidden_units.size(); i++)
			hidden_units[i] = sample_sigmoid_function(hidden_units[i] + hidden_biases[i], generator_for_hidden_sampling);
	}

	inline void sample_hidden_units(my_vector<float>& hidden_units, const my_vector<float>& hidden_biases,
			const uint n_total_units, const uint index)
	{
		skip_gen(generator_for_hidden_sampling, index);

		for(uint i = 0; i != hidden_units.size(); i++)
			hidden_units[i] = sample_sigmoid_function(hidden_units[i] + hidden_biases[i], generator_for_hidden_sampling);

		skip_gen(generator_for_hidden_sampling, n_total_units - index - hidden_units.size());
	}

	inline void reconstruct_visible_units(my_vector<float>& rec_visible_units, const my_vector<float>& visible_biases,
			const bool first_layer)
	{
		//for the first layer we apply gaussian noise
		if(first_layer)
			for(uint i = 0; i != rec_visible_units.size(); i++)
				rec_visible_units[i] =	sigmoid(sample_gaussian_noise(rec_visible_units[i] + visible_biases[i], generator_for_visible_rec_noise));
		else
			for(uint i = 0; i != rec_visible_units.size(); i++)
				rec_visible_units[i] = 	sigmoid(rec_visible_units[i] + visible_biases[i]);
	}


	inline void reconstruct_visible_units(my_vector<float>& rec_visible_units, const my_vector<float>& visible_biases,
			const bool first_layer, const uint n_total_units, const uint index)
	{
		//for the first layer we apply gaussian noise
		if(first_layer)
		{
			skip_gaussian_gen(generator_for_visible_rec_noise, index);

			for(uint i = 0; i != rec_visible_units.size(); i++)
				rec_visible_units[i] =	sigmoid(sample_gaussian_noise(rec_visible_units[i] + visible_biases[i], generator_for_visible_rec_noise));

			skip_gaussian_gen(generator_for_visible_rec_noise, (n_total_units - index - rec_visible_units.size()));
		}
		else
		{
			for(uint i = 0; i != rec_visible_units.size(); i++)
				rec_visible_units[i] = 	sigmoid(rec_visible_units[i] + visible_biases[i]);
		}
	}



	inline void reconstruct_hidden_units(my_vector<float>& rec_hidden_units, const my_vector<float> &hidden_biases,
			const bool first_layer)
	{
		//for the first layer we sample the hidden units
		if(first_layer)
		{
			for(uint i = 0; i != rec_hidden_units.size(); i++)
				rec_hidden_units[i] = sample_sigmoid_function(rec_hidden_units[i] + hidden_biases[i], generator_for_hidden_rec_sampling);
		}
		else
		{
			for(uint i = 0; i != rec_hidden_units.size(); i++)
				rec_hidden_units[i] = sigmoid(rec_hidden_units[i] + hidden_biases[i]);
		}
	}


	inline void reconstruct_hidden_units(my_vector<float>& rec_hidden_units, const my_vector<float> &hidden_biases,
			const bool first_layer, const uint n_total_units, const uint index)
	{
		//for the first layer we sample the hidden units
		if(first_layer)
		{
			skip_gen(generator_for_hidden_rec_sampling, index);

			for(uint i = 0; i != rec_hidden_units.size(); i++)
				rec_hidden_units[i] = sample_sigmoid_function(rec_hidden_units[i] + hidden_biases[i], generator_for_hidden_rec_sampling);

			skip_gen(generator_for_hidden_rec_sampling, n_total_units - index - rec_hidden_units.size());
		}
		else
		{
			for(uint i = 0; i != rec_hidden_units.size(); i++)
				rec_hidden_units[i] = sigmoid(rec_hidden_units[i] + hidden_biases[i]);
		}
	}
}



#endif /* GENERATORS_H_ */
