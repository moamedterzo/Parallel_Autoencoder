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
        int n_hidden_units;
        int n_visible_units;

        float learning_rate;
        float momentum;

        int n_training_epocs;
        int size_minibatch;

        samples_manager samplesmanager;
        std::default_random_engine generator;

        float initial_weights_variance;
        float initial_weights_mean;
        float initial_biases_value;

    public:    

        rbm(bool _first_layer, int _n_visible_units, int _n_hidden_units,
            samples_manager& _samples_manager, std::default_random_engine& _generator);


        inline void update_parameters(
            vector<vector<float>> &weights, 
            vector<float> &hidden_biases, 
            vector<float> &visible_biases,
            vector<vector<float>> &diff_weights,
            vector<float> &diff_visible_biases,
            vector<float> &diff_hidden_biases,
            const int number_of_samples
            );


        void learn(vector<vector<float>>& weights, vector<float>& hidden_biases,
                vector<float>& visible_biases);

        void forward_pass(const vector<float>& input, vector<float>& output,
            const vector<vector<float>>& weights, const vector<float>& hidden_biases);
    };

}



#endif /* RBM_H */

