/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   main.cpp
 * Author: giovanni
 *
 * Created on 14 novembre 2020, 09:33
 */

#include <cstdlib>
#include <iostream>

#include "custom_utils.h"
#include "samples_manager.h"
#include "autoencoder.h"

using namespace std;
using namespace parallel_autoencoder;


/*
 *
 */
int main(int argc, char** argv) {


    try
    {
        std::default_random_engine generator;
        samples_manager sss = samples_manager("./mnist_chinese/data", 4); //todo sistemare

        vector<int> layer_sizes = { 4096 , 2, 4096, 2048, 1024, 512 , 256, 128, 64, 32};

        Autoencoder autoen = Autoencoder(layer_sizes, sss, generator);
        //autoen.load_parameters();

        autoen.Train();


        string path_to_save = "./autoencoder_pars/saved_pars.txt";
        autoen.save_parameters(path_to_save);


        //proviamo il reconstruct
        sss.restart();
        vector<float> input_buffer(layer_sizes[0]);
        while(sss.get_next_sample(input_buffer)){

            auto reconstructed = autoen.reconstruct(input_buffer);

            std::cout << "Root squared error: " << root_squared_error(input_buffer, reconstructed) << "\n";

            std::cout << "Input vector\n";
            //print_vector(input_buffer);
            sss.show_sample(input_buffer);

            std::cout << "Output vector\n";
            //print_vector(reconstructed);
            sss.show_sample(reconstructed);

            //getchar();
        }
    }
    catch(exception e)
    {
        cout << e.what();
    }


    return 0;
}

