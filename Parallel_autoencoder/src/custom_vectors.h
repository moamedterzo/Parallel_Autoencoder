/*
 * custom_vectors.h
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#ifndef CUSTOM_VECTORS_H_
#define CUSTOM_VECTORS_H_


#include <iostream>
#include <random>
#include <cassert>
#include <initializer_list>



namespace parallel_autoencoder
{

	enum class GridOrientation { row_first, col_first };


	template<typename Arg>
	class my_vector
	{
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


	template<typename Arg>
	class matrix
	{
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

				return mem[r * cols + c];
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



	inline float sigmoid(const float x){
		return 1.0 / (1.0 + exp(-x));
	}


	inline float logit(const float p){
		return logf(p / (1-p));
	}






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
				   dest_matrix.at(c, r) = { source_matrix.at(r, c) };
	   }


		inline void print_vector_norm(my_vector<float>& vec, std::string&& myid)
		{
			float result = 0;

			for(uint i = 0; i != vec.size(); i++)
				result +=vec[i];


			std::cout << "[" << myid << ": " << result << "]\n";
		}





		 //Implementa il campionamento basato sulla funzione sigmoide
		 //utilizza un metodo più efficente per il sampling
		inline  float sample_sigmoid_function(const float sigmoid_argument, std::default_random_engine& generator){

			//distribuzione uniforme tra 0 e 1
			static std::uniform_real_distribution<float> uniform_dis(0.0, 1.0);

			auto logit_v = logit(uniform_dis(generator));
			return sigmoid_argument > logit_v ? 1.0 : 0.0;
		}


		 inline float sample_gaussian_distribution(const float mean, const float variance,  std::default_random_engine& generator){

			static std::normal_distribution<float> dist(0, 1.0);

			//imposto parametri
			std::normal_distribution<float>::param_type param(mean, variance);
			dist.param(param);

			return dist(generator);
		}

		 inline float sample_gaussian_distribution(const float mean, std::default_random_engine& generator){

			return sample_gaussian_distribution(mean, 1.0, generator);
		}

		 inline float root_squared_error(my_vector<float>& vec_1, my_vector<float>& vec_2)
		{
			assert(vec_1.size() == vec_2.size());

			float res = 0;

			for(uint i = 0; i != vec_1.size(); i++)
				res += pow(vec_1[i] - vec_2[i], 2);

			return sqrt(res);
		}


		 inline void print_vector(my_vector<float>& v) {

			for(uint i = 0; i != v.size(); i++)
				std::cout << "[" << std::to_string(v[i]) << "] ";

			std::cout << "\n";
		}

		 inline 	void print_matrix(matrix<float>& v) {
			 for(uint i = 0; i != v.get_rows(); i++)
			 {
				 for(uint j = 0; j != v.get_cols(); j++)
					 std:: cout << "[" << v.at(i, j) << "] ";

				 std::cout << "\n";
			 }

			 std::cout << "\n";
		}



		inline void initialize_weight_matrix(matrix<float>& weights,
				const float rbm_initial_weights_mean, const float rbm_initial_weights_variance,
				std::default_random_engine& generator)
		{
			for(uint i = 0; i != weights.size(); i++)
					weights[i] = sample_gaussian_distribution(rbm_initial_weights_mean, rbm_initial_weights_variance, generator);
		}



		inline void SampleHiddenUnits(my_vector<float>& hidden_units, my_vector<float>& hidden_biases, std::default_random_engine& generator)
		{
			for(uint i = 0; i != hidden_units.size(); i++)
				hidden_units[i] = sample_sigmoid_function(hidden_units[i] + hidden_biases[i], generator);
		}

		inline void ReconstructVisibileUnits(my_vector<float>& rec_visible_units, my_vector<float>& visible_biases, const bool first_layer, std::default_random_engine& generator)
		{
			//todo assicurarsi che siano probabilita e non binari
			if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
				for(uint i = 0; i != rec_visible_units.size(); i++)
					rec_visible_units[i] =	sample_gaussian_distribution(rec_visible_units[i] + visible_biases[i], generator);
			else
				for(uint i = 0; i != rec_visible_units.size(); i++)
					rec_visible_units[i] =	sigmoid(rec_visible_units[i] + visible_biases[i]);
		}

		inline void ReconstructHiddenUnits(my_vector<float>& rec_hidden_units, my_vector<float> &hidden_biases, const bool first_layer, std::default_random_engine& generator)
		{
			if(first_layer)
				for(uint i = 0; i != rec_hidden_units.size(); i++)
					rec_hidden_units[i] = sample_sigmoid_function(rec_hidden_units[i] + hidden_biases[i], generator);
			else
				for(uint i = 0; i != rec_hidden_units.size(); i++)
					rec_hidden_units[i] =  sigmoid(rec_hidden_units[i] + hidden_biases[i]);
		}



		inline void apply_sigmoid_to_layer(my_vector<float>& output, my_vector<float> biases, const bool round_output)
		{
			if(round_output)
				for(uint i = 0; i != output.size(); i++)
					output[i] = round(sigmoid(output[i] + biases[i]));
			else
				for(uint i = 0; i != output.size(); i++)
					output[i] = sigmoid(output[i] + biases[i]);
		}


		inline void deltas_for_output_layer(my_vector<float>& output_layer,my_vector<float>& first_activation_layer,my_vector<float>& current_deltas)
		{
			for(uint j = 0; j != output_layer.size(); j++)
				   current_deltas[j] = output_layer[j]  * (1 - output_layer[j]) //derivative
					   * (first_activation_layer[j] - output_layer[j]); //rec error
		}




		//dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
		//la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
		 inline void update_parameters_biases(
						const float momentum,
						const float current_learning_rate,
						my_vector<float> &hidden_biases, my_vector<float> &visible_biases,
						my_vector<float> &diff_visible_biases,	my_vector<float> &diff_hidden_biases,
						const int number_of_samples)
			 {
					//si precalcola il fattore moltiplicativo
					//dovendo fare una media bisogna dividere per il numero di esempi
					const float mult_factor = current_learning_rate / number_of_samples;

					//diff per pesi e bias visibili
					for(uint i = 0; i != visible_biases.size(); i++)
					{
						visible_biases[i] += diff_visible_biases[i] * mult_factor;

						//inizializzazione per il momentum
						diff_visible_biases[i] = diff_visible_biases[i] * momentum;
					}

					for(uint j = 0; j != hidden_biases.size(); j++){
						hidden_biases[j] += diff_hidden_biases[j]* mult_factor;

						//inizializzazione per il momentum
						diff_hidden_biases[j] = diff_hidden_biases[j] * momentum;
					}
			}


		 inline void update_parameters_weights(
					const float momentum,
					const float current_learning_rate,
					matrix<float>& weights, matrix<float>& diff_weights,
					const int number_of_samples)
			{
					//si precalcola il fattore moltiplicativo
					//dovendo fare una media bisogna dividere per il numero di esempi
					const float mult_factor = current_learning_rate / number_of_samples;

					//diff per pesi visibili
					for(uint i = 0; i < weights.size(); i++)
					{
					   weights[i] += diff_weights[i] * mult_factor;

					   //inizializzazione per il momentum
					   diff_weights[i] = diff_weights[i] * momentum;
					}
			}


		 inline void update_weights_fine_tuning(matrix<float>& weights_to_update,
							my_vector<float>& deltas,my_vector<float>& input_layer,
							const float fine_tuning_learning_rate)
		{
			assert( weights_to_update.get_rows() == input_layer.size());
			assert( weights_to_update.get_cols() == deltas.size());

			for(uint i = 0; i != weights_to_update.get_rows(); i++){
				for(uint j = 0; j != weights_to_update.get_cols(); j++){
					//delta rule
					weights_to_update.at(i, j) +=
							fine_tuning_learning_rate
							* deltas[j]
							* input_layer[i];
				}
			}
		}



			inline void update_biases_fine_tuning(my_vector<float>& biases_to_update,my_vector<float>& current_deltas,
					const float fine_tuning_learning_rate)
			{
				assert(biases_to_update.size() == current_deltas.size());

				for(uint j = 0; j != biases_to_update.size(); j++)
					 biases_to_update[j] += fine_tuning_learning_rate * current_deltas[j];
			}




	inline void print_sec_mpi(std::ostream& buf, double t0, double t1, int myid)
	{
		buf << "\nTotal time (MPI) " + std::to_string(myid) + " is " + std::to_string(t1 - t0) + "\n";
	}

	inline void print_sec_gtd(std::ostream& buf, timeval& wt0, timeval& wt1, int myid)
	{
		long sec  = (wt1.tv_sec  - wt0.tv_sec);
		long usec = (wt1.tv_usec - wt0.tv_usec);

		if(usec < 0) {
			--sec;
			usec += 1000000;
		}

		auto diffsec  = ((double)(sec*1000)+ (double)usec/1000.0);

		buf << "total time (gtd) " + std::to_string(myid) +  " is " + std::to_string(diffsec) + "\n";
	}

}




#endif /* CUSTOM_VECTORS_H_ */
