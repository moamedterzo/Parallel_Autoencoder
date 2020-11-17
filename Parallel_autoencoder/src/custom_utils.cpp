/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "custom_utils.h"


#include <vector>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
#include <cassert>   

using std::cout;
using std::vector;


namespace parallel_autoencoder{
    
    /*void matrix_multiplication(const vector<vector<float>> m_n, 
            const vector<vector<float>> n_r, 
            vector<vector<float>> m_r)
    {
       int i, j, k;
       int m = m_n.size();
       int r = n_r[0].size();
       m_r.reserve(m);
       
       for(i = 0; i < m; ++i) //per ciascuna riga
       {       
          m_r[i].reserve(r);
          for(j = 0; j < r; ++j) //per ciascuna colonna       
             for(k = 0; k < n_r.size(); ++k)        
                 m_r[i][j] += m_n[i][k] * n_r[k][j];        
       }
    }*/
    
    //questo metodo riserva spazio per il vettore destinazione
    void matrix_transpose(const vector<vector<float>>& source_mat, vector<vector<float>>& dest_mat){
        
        int rows = source_mat.size();
        int cols = source_mat[0].size();
        
        for(int i = 0; i < cols; i++)
        {
            for(int j = 0; j < rows; j++)
            {
                dest_mat[i][j] = source_mat[j][i];
            }
        }        
    }
    
    /*void map_function_vector(const vector<float> source_vec, vector<float> dest_vec, float map(float)){
        assert(source_vec.size() == dest_vec.size());
        
        for(int i = 0; i < source_vec.size(); i++)
            dest_vec[i] = map(source_vec[i]);        
    }*/
    
 
    void matrix_vector_multiplication(const vector<vector<float>>& m_by_n_matrix, 
            const vector<float>& n_by_1_vec, 
            const vector<float>& m_by_1_vect_bias,
            vector<float>& m_by_1_vec_dest)
    {
        int m = m_by_n_matrix.size();
        int n = m_by_n_matrix[0].size();
        
        assert(n == n_by_1_vec.size()); //si controlla se le grandezze corrispondono
        assert(m == m_by_1_vect_bias.size()); //si controlla se le grandezze corrispondono
        assert(m == m_by_1_vec_dest.size()); //si controlla se le grandezze corrispondono
        
        //m_by_1_vec_dest.reserve(m);
        
        //moltiplicazione matrice per vettore
        for(int i = 0; i < m; i++)
        {
            m_by_1_vec_dest[i] = m_by_1_vect_bias[i];
            for(int j = 0; j < n; j++)
                m_by_1_vec_dest[i] += m_by_n_matrix[i][j] * n_by_1_vec[j];
        }
    }
    
    //metodo creato per non dover effettuare di volta in volta la trasposta della matrice
    void matrix_transpose_vector_multiplication(const vector<vector<float>>& m_by_n_matrix, 
            const vector<float>& m_by_1_vec, 
            const vector<float>& n_by_1_vect_bias,
            vector<float>& n_by_1_vec_dest)
    {
        int m = m_by_n_matrix.size();
        int n = m_by_n_matrix[0].size();
        
        assert(m == m_by_1_vec.size()); //si controlla se le grandezze corrispondono
        assert(n == n_by_1_vect_bias.size()); //si controlla se le grandezze corrispondono
        assert(n == n_by_1_vec_dest.size()); //si controlla se le grandezze corrispondono
        
        //n_by_1_vec_dest.reserve(n);
        
        //moltiplicazione matrice trasposta per vettore
        for(int j = 0; j < n; j++)
        {
            n_by_1_vec_dest[j] = n_by_1_vect_bias[j];
            for(int i = 0; i < m; i++)
                n_by_1_vec_dest[j] += m_by_n_matrix[i][j] * m_by_1_vec[i];
        }
    }
    
  /*  void vector_vector_multiplication(
    const vector<float> m_by_1_vect, const vector<float> n_by_1_vect,
            vector<vector<float>> m_by_n_matrix){
    
        for(int i = 0; i < m_by_1_vect; i++)
            for(int j = 0; j < n_by_1_vect; j++)
                m_by_n_matrix[i][j] += m_by_1_vect[i] * n_by_1_vect[j]; //si effettua un'aggiunta
    }*/
    
    
    /*void accumulate_sum_into_vector(const vector<float>& source_vec, vector<float>& dest_vec){
    
        assert(source_vec.size() == dest_vec.size());
        
        for(int i = 0; i < source_vec.size(); i++)
            dest_vec[i] += source_vec[i];
    }*/
    
    float sigmoid(const float x){
        return 1.0 / (1.0 + exp(-x)); 
    }
    
    
    float logit(const float p){
        return logf(p / (1-p));
    }
    
    //Implementa il campionamento basato sulla funzione sigmoide
    //utilizza un metodo più efficente per il sampling
    float sample_sigmoid_function(const float sigmoid_argument, std::default_random_engine& generator){
        
        //distribuzione uniforme tra 0 e 1
        static std::uniform_real_distribution<float> uniform_dis(0.0, 1.0);
                
        return sigmoid_argument > logit(uniform_dis(generator)) ? 1.0 : 0.0;
    }
    
    //Implementa la generazione di un numero da una distribuzione gaussiana con media variabile e varianza unitaria
        float sample_gaussian_distribution(const float mean, const float variance, 
            std::default_random_engine& generator){
                
        static std::normal_distribution<float> dist(0, 1.0);
        
        //imposto parametri
        std::normal_distribution<float>::param_type param(mean, variance);
        dist.param(param);       
                
        return dist(generator);
    }  
    
    float sample_gaussian_distribution(const float mean, 
            std::default_random_engine& generator){
                
        return sample_gaussian_distribution(mean, 1.0, generator);
    }        

    
    float root_squared_error(vector<float> vec_1, vector<float> vec_2)
    {
        assert(vec_1.size() == vec_2.size());
        
        float res = 0;
        
        for(int i = 0; i < vec_1.size(); i++)
            res += pow(vec_1[i] - vec_2[i], 2);
        
        return sqrt(res);
    }   
    
    
    void print_vector(vector<float> v) {
        for (auto elem : v) {
            cout << "[" << elem << "] ";
        }
        
        
        cout <<  "\n";
    }
    
    void print_matrix(vector<vector<float>> v) {
        for (auto elem : v) {
            print_vector(elem);
        }
        
        cout << "\n";
    }

}

