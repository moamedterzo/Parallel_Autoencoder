/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   rbm.h
 * Author: giovanni
 *
 * Created on 14 novembre 2020, 09:58
 */

#ifndef RBM_H
#define RBM_H

#include <array>
#include <random>

#include "samples_manager.h"


namespace parallel_autoencoder{

    class rbm{

    private:

        bool first_layer;
        uint n_hidden_units;
        uint n_visible_units;

        float learning_rate;
        float momentum;

        uint n_training_epocs;
        uint size_minibatch;

        samples_manager samplesmanager;
        std::default_random_engine generator;

        float initial_weights_variance;
        float initial_weights_mean;
        float initial_biases_value;

    public:    

        rbm(bool _first_layer, uint _n_visible_units, uint _n_hidden_units,
            samples_manager& _samples_manager, std::default_random_engine& _generator);


        inline void update_parameters(
            matrix<float> &weights,
            my_vector<float> &hidden_biases, my_vector<float> &visible_biases,
			matrix<float> &diff_weights,
			my_vector<float> &diff_visible_biases, my_vector<float> &diff_hidden_biases,
            const uint number_of_samples
            );


        void learn(matrix<float>& weights, my_vector<float>& hidden_biases, my_vector<float>& visible_biases);

        void forward_pass(const my_vector<float>& input, my_vector<float>& output,
            const matrix<float>& weights, const my_vector<float>& hidden_biases);
    };

}



#endif /* RBM_H */

