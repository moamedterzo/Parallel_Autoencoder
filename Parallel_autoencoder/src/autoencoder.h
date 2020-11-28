/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   autoencoder.h
 * Author: giovanni
 *
 * Created on 14 novembre 2020, 09:46
 */



#ifndef AUTOENCODER_H
#define AUTOENCODER_H


#include <vector>
#include <random>

#include "samples_manager.h"

using std::vector;


namespace parallel_autoencoder{

    class Autoencoder
    {
    private:
        my_vector<uint> layers_size;
        uint number_of_rbm_to_learn;
        uint number_of_final_layers;

        my_vector<matrix<float>> layers_weights;
        my_vector<my_vector<float>> layer_biases;

        float fine_tuning_learning_rate;
        float fine_tuning_n_training_epocs;

        samples_manager samplesmanager;
        std::default_random_engine generator;

        bool fine_tuning_finished;
        uint trained_rbms;

        string parameters_tosave_file_path = "./autoencoder_pars/single_node.txt";

    public:
        Autoencoder(const my_vector<uint>& _layers_size, samples_manager& _samplesmanager,
                std::default_random_engine& _generator);

        void set_size_for_layers(const my_vector<uint>& _layers_size_source);

        void Train();

        my_vector<bool> encode(my_vector<float> input);
        my_vector<float> reconstruct(my_vector<float> input);

        void save_parameters();
        void save_parameters(string& path_file);
        void load_parameters();
        void load_parameters(string& path_file);
    };
}


#endif /* AUTOENCODER_H */

