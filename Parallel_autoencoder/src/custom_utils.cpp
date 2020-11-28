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
        

	void print_vector_norm(my_vector<float>& vec, std::string&& myid)
	{
		float result = 0;

		for(uint i = 0; i != vec.size(); i++)
			result +=vec[i];


		std::cout << "[" << myid << ": " << result << "]\n";
	}
 

	 void matrix_vector_multiplication(const matrix<float>& m_by_n_matrix,
			const my_vector<float>& n_by_1_vec,
			const my_vector<float>& m_by_1_vect_bias,
			my_vector<float>& m_by_1_vec_dest)
    {
		//si controlla se le grandezze corrispondono
        assert(m_by_n_matrix.get_cols() == n_by_1_vec.size());
        assert(m_by_n_matrix.get_rows() == m_by_1_vect_bias.size());
        assert(m_by_n_matrix.get_rows() == m_by_1_vec_dest.size());

        //moltiplicazione matrice per vettore
        for(uint i = 0; i != m_by_n_matrix.get_rows(); i++)
        {
            m_by_1_vec_dest[i] = m_by_1_vect_bias[i];
            for(uint j = 0; j != m_by_n_matrix.get_cols(); j++)
                m_by_1_vec_dest[i] += m_by_n_matrix.at(i, j)  * n_by_1_vec[j];
        }
    }
    
	 void matrix_transpose_vector_multiplication(const matrix<float>& m_by_n_matrix,
	 			const my_vector<float>& m_by_1_vec,
	 			const my_vector<float>& n_by_1_vect_bias,
	 			my_vector<float>& n_by_1_vec_dest)
    {
		 //si controlla se le grandezze corrispondono
        assert(m_by_n_matrix.get_rows() == m_by_1_vec.size());
        assert(m_by_n_matrix.get_cols() == n_by_1_vect_bias.size());
        assert(m_by_n_matrix.get_cols() == n_by_1_vec_dest.size());
        
        //n_by_1_vec_dest.reserve(n);
        
        //moltiplicazione matrice trasposta per vettore
        for(uint j = 0; j != m_by_n_matrix.get_cols(); j++)
        {
            n_by_1_vec_dest[j] = n_by_1_vect_bias[j];
            for(uint i = 0; i != m_by_n_matrix.get_rows(); i++)
                n_by_1_vec_dest[j] += m_by_n_matrix.at(i, j) * m_by_1_vec[i];
        }
    }

    
    float sigmoid(const float x){
        return 1.0 / (1.0 + exp(-x)); 
    }
    
    
    float logit(const float p){
        return logf(p / (1-p));
    }
    

    float sample_sigmoid_function(const float sigmoid_argument, std::default_random_engine& generator){
        
        //distribuzione uniforme tra 0 e 1
        static std::uniform_real_distribution<float> uniform_dis(0.0, 1.0);
                
        auto logit_v = logit(uniform_dis(generator));
        return sigmoid_argument > logit_v ? 1.0 : 0.0;
    }
    

    float sample_gaussian_distribution(const float mean, const float variance,  std::default_random_engine& generator){
                
        static std::normal_distribution<float> dist(0, 1.0);
        
        //imposto parametri
        std::normal_distribution<float>::param_type param(mean, variance);
        dist.param(param);       
                
        return dist(generator);
    }  
    
    float sample_gaussian_distribution(const float mean, std::default_random_engine& generator){
                
        return sample_gaussian_distribution(mean, 1.0, generator);
    }        

    
    float root_squared_error(my_vector<float> vec_1, my_vector<float> vec_2)
    {
        assert(vec_1.size() == vec_2.size());
        
        float res = 0;
        
        for(uint i = 0; i != vec_1.size(); i++)
            res += pow(vec_1[i] - vec_2[i], 2);
        
        return sqrt(res);
    }   
    
    
    void print_vector(my_vector<float> v) {

        for(uint i = 0; i != v.size(); i++)
            cout << "[" << std::to_string(v[i]) << "] ";

        cout << "\n";
    }
    
    void print_matrix(matrix<float> v) {
    	 for(uint i = 0; i != v.get_rows(); i++)
    	 {
    		 for(uint j = 0; j != v.get_cols(); j++)
    			 cout << "[" << v.at(i, j) << "] ";

    		 cout << "\n";
    	 }

        cout << "\n";
    }

}

