/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   samples_manager.h
 * Author: giovanni
 *
 * Created on 14 novembre 2020, 10:10
 */

#ifndef SAMPLES_MANAGER_H
#define SAMPLES_MANAGER_H

#include <string>
#include <vector>
#include <dirent.h>

using std::string;
using std::vector;

namespace parallel_autoencoder{

    class samples_manager{
    private:
        DIR *dp = nullptr; 
        int height = 0;
        int width = 0;
        int current_sample_number = 0;

    public:

        int max_n_samples = 0;
        string path_folder = "";

        samples_manager();
        samples_manager(string _path_folder, int _max_n_samples);


        void init();

        void restart();

        uint get_number_samples();
        dirent* get_next_dir();

        bool get_next_sample(vector<float>& buffer, string *filename);

        bool get_next_sample(vector<float>& buffer);



        void save_sample(vector<float>& buffer, string folder, string filepath);
        
        void show_sample(vector<float>& buffer);

        void close();

    };
}

#endif /* SAMPLES_MANAGER_H */

