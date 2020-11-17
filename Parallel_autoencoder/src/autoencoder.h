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
        vector<int> layers_size;
        int number_of_rbm_to_learn;
        int number_of_final_layers;

        vector<vector<vector<float>>> layers_weights;
        vector<vector<float>> layer_biases;

        float fine_tuning_learning_rate;
        float fine_tuning_n_training_epocs;

        samples_manager samplesmanager;
        std::default_random_engine generator;

        bool fine_tuning_finished;
        int trained_rbms;

        string parameters_tosave_file_path = "./autoencoder_pars/temp.txt";    

    public:
        Autoencoder(const vector<int>& _layers_size, samples_manager& _samplesmanager, 
                std::default_random_engine _generator);

        void Train();

        vector<bool> encode(vector<float> input);
        vector<float> reconstruct(vector<float> input);

        void save_parameters();
        void save_parameters(string& path_file);
        void load_parameters();
        void load_parameters(string& path_file);

    };
}


#endif /* AUTOENCODER_H */

