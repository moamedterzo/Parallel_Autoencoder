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
#include "custom_utils.h"

using std::string;
using std::vector;

namespace parallel_autoencoder{

	//todo capire che numero mettere
	static const int F_PREC = 9;

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
        dirent* get_next_dir(const char* extension);


        bool get_next_sample(my_vector<float>& buffer, const char* extension, string *filename);
        bool get_next_sample(my_vector<float>& buffer, const char* extension);


        void save_sample(my_vector<float>& buffer,bool save_as_image,  string folder, string filepath);

        void show_sample(my_vector<float>& buffer);



        void close();

    };
}

#endif /* SAMPLES_MANAGER_H */

