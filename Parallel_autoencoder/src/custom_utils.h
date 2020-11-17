/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   utils.h
 * Author: giovanni
 *
 * Created on 14 novembre 2020, 09:47
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>   

using std::vector;


namespace parallel_autoencoder{
    
    
    void matrix_transpose(const vector<vector<float>>& source_mat, vector<vector<float>>& dest_mat);
    
 
    void matrix_vector_multiplication(const vector<vector<float>>& m_by_n_matrix, 
            const vector<float>& n_by_1_vec, 
            const vector<float>& m_by_1_vect_bias,
            vector<float>& m_by_1_vec_dest);
    
    //metodo creato per non dover effettuare di volta in volta la trasposta della matrice
    void matrix_transpose_vector_multiplication(const vector<vector<float>>& m_by_n_matrix, 
            const vector<float>& m_by_1_vec, 
            const vector<float>& n_by_1_vect_bias,
            vector<float>& n_by_1_vec_dest);
    
    float sigmoid(const float x);
    
    
    float logit(const float p);
    
    //Implementa il campionamento basato sulla funzione sigmoide
    //utilizza un metodo più efficente per il sampling
    float sample_sigmoid_function(const float sigmoid_argument, std::default_random_engine& generator);
    
    //Implementa la generazione di un numero da una distribuzione gaussiana con media variabile e varianza unitaria
    float sample_gaussian_distribution(const float mean, const float variance, 
            std::default_random_engine& generator);
    
    float sample_gaussian_distribution(const float mean, 
            std::default_random_engine& generator);

    
    float root_squared_error(vector<float> vec_1, vector<float> vec_2);
    
    
    void print_vector(vector<float> v) ;
    
    void print_matrix(vector<vector<float>> v);

}




#endif /* UTILS_H */

