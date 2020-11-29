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

#include <cstring>
#include <cassert>
#include "mpi.h"
#include <iostream>
#include <initializer_list>

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

		~my_vector()
		{
			if(mem)
				delete[] mem;
		}


		my_vector& operator=(my_vector&&  tmp)
		{
			if(mem)
			  delete [] mem;

			std::swap(mem_size, tmp.mem_size);
			std::swap(mem, tmp.mem);

			tmp.mem = nullptr;

			return *this;
		}

		my_vector(my_vector&& tmp)
		{
			std::swap(mem_size, tmp.mem_size);
			std::swap(mem, tmp.mem);

			tmp.mem = nullptr;
		}

		my_vector& operator=(const my_vector& rhs)
		{
			if( this != &rhs )
			{
				if(mem)
				  delete [] mem;

				mem = new Arg[ rhs.mem_size ];
				for(uint i = 0; i < rhs.mem_size; ++i )
				  mem[i] = rhs.mem[ i ];

				mem_size = rhs.mem_size;
			}

			return *this;
		}


		my_vector(const my_vector& rhs)
		{
			mem = new Arg[ rhs.mem_size ];
			for(uint i = 0; i < rhs.mem_size; ++i )
			  mem[i] = rhs.mem[ i ];

			mem_size = rhs.mem_size;
		}



		my_vector()
		{
			mem_size = 0;
			mem = nullptr;
		}


		my_vector(const uint s)
		{
			assert(s>= 0);

			mem_size = s;
			mem = new Arg[s];
		}


		my_vector(const uint s, const Arg&& init_v) : my_vector(s)
		{
			//valore iniziale
			for(uint i = 0; i != mem_size; i++)
				mem[i] = init_v;
		}

		my_vector(const uint s, const Arg& init_v) : my_vector(s)
		{
			//valore iniziale
			for(uint i = 0; i != mem_size; i++)
				mem[i] = init_v;
		}

		my_vector(std::initializer_list<Arg> lst) : my_vector(lst.size())
		{
			uint i = 0;
			for(auto& v : lst)
				mem[i++] = v;
		}


		void push_back(Arg value)
		{
			//versione inefficiente, meglio evitarla
			Arg *old_mem = mem;
			std::swap(old_mem, mem);

			//copia valori nel nuovo puntatore
			mem = new Arg[mem_size + 1];
			if(old_mem != nullptr)
				for(uint i = 0; i < mem_size; i++)
				    mem[i] = old_mem[i];

			mem[mem_size] = value;

			//nuova size
			mem_size++;

			//si elimina il vecchio puntatore e si aggiorna il corrente
			if(old_mem != nullptr)
				delete[] old_mem;
		}


		Arg& operator[](const uint s) const
		{
			assert(s < mem_size);

			return mem[s];
		}

		Arg* data() const
		{
			return mem;
		}


		uint size() const
		{
			return mem_size;
		}



	};





	struct MPReqManager
	{
		MPReqManager(MPI_Request *reqs,my_vector<MP_Comm_MasterSlave> *comms)
		{
			this->comms = comms;
			this->reqs = reqs;
		}


		const MPI_Datatype mpi_datatype_tosend = MPI_FLOAT;

		MPI_Request *reqs;
		my_vector<MP_Comm_MasterSlave> *comms;

		//virtual void SendVectorToReduce(my_vector<float>&) = 0;

		//virtual void ReceiveVector(my_vector<float>&) = 0;


		void wait()
		{
			MPI_Waitall(comms->size(), reqs, MPI_STATUSES_IGNORE);
		}


		virtual ~MPReqManager(){}
	};


	struct MPReqManagerCell : MPReqManager
	{
		MPReqManagerCell(MPI_Request *reqs, my_vector<MP_Comm_MasterSlave> *comms)
		: MPReqManager{reqs, comms }
		{}


		void SendVectorToReduce(my_vector<float>& vec)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(vec.data() + displacement, MPI_IN_PLACE,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm , reqs + i);


				displacement += comm.n_items_to_send;
			}
		}


		void ReceiveVector(my_vector<float>& vec)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement, comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm,  reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void ReceiveVectorSync(my_vector<float>& vec)
		{
			ReceiveVector(vec);
			wait();
		}

		~MPReqManagerCell(){}
	};


	struct MPReqManagerAccumulator : MPReqManager
	{
		MPReqManagerAccumulator(MPI_Request *reqs, my_vector<MP_Comm_MasterSlave> *comms)
		: MPReqManager{reqs, comms }
		{}


		void BroadcastVector(my_vector<float>& vec)
		{
			//Si invia il vettore alle righe/colonne di riferimento, per ognuna si usa il broadcast
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void AccumulateVector(my_vector<float>& vec)
		{
			int displacement = 0;

			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(MPI_IN_PLACE, vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}


		void BroadcastVectorSync(my_vector<float>& vec)
		{
			BroadcastVector(vec);
			wait();
		}

		void AccumulateVectorSync(my_vector<float>& vec)
		{
			AccumulateVector(vec);
			wait();
		}



		~MPReqManagerAccumulator(){}
	};







	template<typename Arg>
	class matrix{
	private:
		uint rows, cols;
		Arg* mem;

	public:

		matrix()
		{
			rows = cols = 0;
			mem = nullptr;
		}


		~matrix()
		{
			if(mem)
				delete[] mem;
		}



		matrix& operator=(matrix&&  tmp)
		{
			if(mem)
				delete [] mem;

			std::swap(cols, tmp.cols);
			std::swap(rows, tmp.rows);
			std::swap(mem, tmp.mem);

			tmp.mem = nullptr;

			return *this;
		}

		matrix(matrix&& tmp)
		{
			std::swap(cols, tmp.cols);
			std::swap(rows, tmp.rows);
			std::swap(mem, tmp.mem);

			tmp.mem = nullptr;
		}

		matrix& operator=(const matrix& rhs)
		{
			if( this != &rhs )
			{
				if(mem)
				  delete [] mem;

				mem = new Arg[rhs.size()];
				for(uint i = 0; i < rhs.size(); ++i )
				  mem[i] = rhs.mem[ i ];

				cols = rhs.get_cols();
				rows = rhs.get_rows();
			}

			return *this;
		}


		matrix(const matrix& rhs)
		{
			mem = new Arg[rhs.size()];
			for(uint i = 0; i < rhs.size(); ++i )
			  mem[i] = rhs.mem[ i ];

			cols = rhs.get_cols();
			rows = rhs.get_rows();
		}



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

		Arg& operator[](const uint s) const
		{
			assert(s < size());

			return mem[s];
		}

		uint size() const
		{
			return cols * rows;
		}

		uint get_cols() const
		{
			return cols;
		}

		uint get_rows() const
		{
			return rows;
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
			m_vec_dest[i] = 0; //inizializzazione
			for(uint j = 0; j != n; j++)
				m_vec_dest[i] += mn_matrix.at(i, j) * n_vec[j];
		}

	}


   template<typename Arg>
   inline void matrix_transpose_vector_multiplication(const matrix<Arg>& mn_matrix,
			const my_vector<Arg>& m_vec,
			my_vector<Arg>& n_vec_dest)
	{
		auto m = mn_matrix.get_rows();
		auto n = mn_matrix.get_cols();

		 //si controlla se le grandezze corrispondono
		assert(n == n_vec_dest.size());
		assert(m == m_vec.size());

		//moltiplicazione matrice per vettore
		for(uint j = 0; j != n; j++)
		{
			n_vec_dest[j] = 0; //inizializzazione

			for(uint i = 0; i != m; i++)
				n_vec_dest[j] += mn_matrix.at(i, j) * m_vec[i];
		}

	}


   template<typename Arg>
   inline void transpose_matrix(const matrix<Arg>& source_matrix, matrix<Arg>& dest_matrix)
   {
	   assert(source_matrix.get_rows() == dest_matrix.get_cols());
	   assert(source_matrix.get_cols() == dest_matrix.get_rows());

	   for(uint r = 0; r != source_matrix.get_rows(); r++)
		   for(uint c = 0; c != source_matrix.get_cols(); c++)
			   dest_matrix.at(c, r) = source_matrix.at(r, c);
   }



	void print_vector_norm(my_vector<float>& vec, std::string&& myid);







	//metodo creato per non dover effettuare di volta in volta la trasposta della matrice
	void matrix_transpose_vector_multiplication(const matrix<float>& m_by_n_matrix,
			const my_vector<float>& m_by_1_vec,
			const my_vector<float>& n_by_1_vect_bias,
			my_vector<float>& n_by_1_vec_dest);


 
    void matrix_vector_multiplication(const matrix<float>& m_by_n_matrix,
            const my_vector<float>& n_by_1_vec,
            const my_vector<float>& m_by_1_vect_bias,
			my_vector<float>& m_by_1_vec_dest);
    
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
    
    
    void print_vector(my_vector<float> v);

    void print_matrix(matrix<float> v);

}




#endif /* UTILS_H */

