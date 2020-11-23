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
#include <cassert>
#include "mpi.h"

using std::vector;


namespace parallel_autoencoder{

	//utilizzata per identificare il processo root nell'insieme
	struct MP_Comm_MasterSlave{

		MPI_Comm comm;
		uint root_id;
		uint row_col_id;
		uint n_items_to_send;
	};



	template<typename Arg>
	class my_vector{

	private:
		uint mem_size;
		Arg* mem;

	public:
		my_vector(const uint s) : mem_size{ s }
		{
			assert(s>= 0);

			mem = new Arg[s];
		}

		my_vector(const uint s, const Arg&& init_v) : my_vector(s)
		{
			//valore iniziale
			for(uint i = 0; i != mem_size; i++)
				mem[i] = init_v;
		}


		Arg& operator[](const uint s) const
		{
			assert(s < mem_size);

			return mem[s];
		}


		uint size() const
		{
			return mem_size;
		}


		~my_vector()
		{
			delete mem;
		}

	};









	template<typename Arg>
	class matrix{
	private:
		uint rows, cols;
		Arg* mem;

	public:
		matrix(const uint r, const uint c) : rows{r}, cols{c}
		{
			assert(r >= 0);
			assert(c>=0);

			mem = new Arg[r * c];
		}

		matrix(const uint r, const uint c, const Arg&& init_v) : matrix(r, c)
		{
			//valore iniziale
			for(uint r = 0; r != rows; r++)
				for(uint c=0; c != cols; c++)
					mem[r * c + c] = init_v;
		}

		Arg& at(const uint r, const uint c) const
		{
			assert(r < rows);
			assert(c < cols);

			return mem[r * c + c];
		}

		uint get_cols() const
		{
			return cols;
		}

		uint get_rows() const
		{
			return rows;
		}



		~matrix()
		{
			delete mem;
		}
	};

	template<typename Arg>
   inline void matrix_vector_multiplication(const matrix<Arg>& mn_matrix,
			const my_vector<Arg>& n_vec,
			my_vector<Arg>& m_vec_dest)
	{
		auto m = mn_matrix.get_rows();
		auto n = mn_matrix.get_cols();

		 //si controlla se le grandezze corrispondono
		assert(n == n_vec.size());
		assert(m == m_vec_dest.size());

		//moltiplicazione matrice per vettore
		for(uint i = 0; i != m; i++)
		{
			m_vec_dest.at(i) = 0; //inizializzazione
			for(uint j = 0; j != n; j++)
				m_vec_dest[i] += mn_matrix.at(i, j) * n_vec[j];
		}

	}

	//metodo creato per non dover effettuare di volta in volta la trasposta della matrice
	void matrix_transpose_vector_multiplication(const vector<vector<float>>& m_by_n_matrix,
			const vector<float>& m_by_1_vec,
			const vector<float>& n_by_1_vect_bias,
			vector<float>& n_by_1_vec_dest);



    
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

