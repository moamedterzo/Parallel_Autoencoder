/*
 * node_autoencoder.cpp
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#include "node_autoencoder.h"

#include <string.h>
#include <sys/time.h>



namespace parallel_autoencoder
{

	  //Questo metodo calcola il numero di elementi che (1) un accumulatore deve scambiare verso
		//i nodi della griglia oppure (2) gli elementi che un nodo della griglia deve scambiare
		//con ciascun accumulatore collegato.
		void node_autoencoder::calc_comm_sizes(const GridOrientation current_or, //orientamento comm
				my_vector<MP_Comm_MasterSlave>& acc_gridcolrow_comm, //comm da accumulatore a righe o colonne
				const bool calc_for_acc, //indica se il calcolo va fatto per l'accumulatore o il nodo della griglia
				const uint my_k_col_row_number, //indice del nodo k o della cella in riga r o colonna c
				const int n_tot_vishid_units)
		{
			auto total_grid_rowcol = (current_or == GridOrientation::row_first ? grid_rows : grid_cols);

			uint index_acc = 0;
			uint index_gridnode = 0;

			uint elements_for_acc = get_units_for_node(n_tot_vishid_units, total_accumulators, index_acc);
			uint elements_for_grid_node = get_units_for_node(n_tot_vishid_units, total_grid_rowcol, index_gridnode);

			do
			{
				if(calc_for_acc)
				{
					//se questo calcolo lo si sta facendo per l'accumulatore...
					if(index_acc == my_k_col_row_number)
					{
						//ora abbiamo ottenuto il numero di elementi che l'accumulatore deve dare verso una determinata riga/colonna
						//si cerca il comunicatore della riga/colonna di riferimento
						for(uint cc = 0; cc != acc_gridcolrow_comm.size(); cc++)
						{
							auto& comm = acc_gridcolrow_comm[cc];
							if(comm.row_col_id == index_gridnode)
							{
								comm.n_items_to_send = std::min(elements_for_acc, elements_for_grid_node);

								//log
								oslog << "Current orientation:" << (current_or == GridOrientation::row_first ? "row" : "col") << "\n";
								oslog << "I'm a k node, I will send " << comm.n_items_to_send << " (over a total of " << n_tot_vishid_units << ")" <<
										" elements to each node of the row/col with index:"  << index_gridnode << "\n";
								oslog.flush();

								break;
							}
						}
					}
				}
				else
				{
					//se questo calcolo lo si sta facendo per il nodo della cella ...
					if(index_gridnode == my_k_col_row_number)
					{
						//ora abbiamo ottenuto il numero di elementi che l'accumulatore deve dare verso una determinata riga/colonna
						//si cerca il comunicatore della riga/colonna di riferimento
						for(uint cc = 0; cc != acc_gridcolrow_comm.size(); cc++)
						{
							auto& comm = acc_gridcolrow_comm[cc];
							if(comm.acc_root_id == index_acc)
							{
								comm.n_items_to_send = std::min(elements_for_acc, elements_for_grid_node);

								//log
								oslog << "Current orientation:" << (current_or == GridOrientation::row_first ? "row" : "col") << "\n";
								oslog << "I'm a cell node, I will send " << comm.n_items_to_send << " (over a total of " << n_tot_vishid_units << ")"
										" elements to the k node with index: " << index_acc  << "\n";
								oslog.flush();

								break;
							}
						}
					}
				}

				int diff = elements_for_acc - elements_for_grid_node;

				if(diff >= 0)
				{
					if(diff > 0)
						elements_for_acc = diff;

					index_gridnode++;
					elements_for_grid_node = get_units_for_node(n_tot_vishid_units, total_grid_rowcol, index_gridnode);
				}

				if(diff <= 0)
				{
					if(diff < 0)
						elements_for_grid_node = diff * -1;

					index_acc++;
					elements_for_acc = get_units_for_node(n_tot_vishid_units, total_accumulators, index_acc);
				}
			}
			while((calc_for_acc ? index_acc : index_gridnode) <= my_k_col_row_number);
			//non mi fermo fintantoché non ho analizzato tutti i possibili collegamenti del nodo
		}


		//restituisce unità (visibili o hidden) per l'accumulatore k o il nodo della riga r o della colonna c
		  int node_autoencoder::get_units_for_node(const uint n_total_units, const uint total_nodes, const uint node_number)
		  {
			const uint n_units_x_node = ceil((float)n_total_units / total_nodes);

			const int overflow_units = (node_number + 1) * n_units_x_node - n_total_units;

			int n_my_units = n_units_x_node;
			if(overflow_units > 0)
			{
				n_my_units -= overflow_units;
				if(n_my_units < 0) n_my_units = 0; //caso limite
			}

			return n_my_units;
		  }


		float node_autoencoder::GetRBMLearningRate(const uint epoch_number, const uint layer_number)
		{
			//il learning rate varia a seconda dell'epoca e del layer da apprendere
			static const float rbm_learning_rate = 0.01;

			if(layer_number == 0)
				return rbm_learning_rate / 10 * (1 + epoch_number * 0.1);
			else
				return rbm_learning_rate / (1 + epoch_number * 0.1);
		}


		node_autoencoder::node_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
		        		uint _total_accumulators, uint _grid_row, uint _grid_col,
						uint rbm_n_epochs, uint finetuning_n_epochs, bool _batch_mode,
						std::ostream& _oslog, int _mpi_rank)

		:oslog{_oslog}
		{
			mpi_rank = _mpi_rank;

			total_accumulators =_total_accumulators;
			grid_rows =_grid_row;
			grid_cols = _grid_col;

			batch_mode = _batch_mode;
			generator = _generator;

			number_of_rbm_to_learn = _layers_size.size() - 1;
			number_of_final_layers = _layers_size.size() * 2 - 1;

			trained_rbms = 0;
			rbm_momentum = 0.9;
			rbm_n_training_epocs = rbm_n_epochs;
			rbm_size_minibatch = 128;

			rbm_initial_weights_variance = 0.01;
			rbm_initial_weights_mean = 0;
			rbm_initial_biases_value = 0;


			fine_tuning_n_training_epocs = finetuning_n_epochs;
			fine_tuning_learning_rate = 10e-6;

			fine_tuning_finished = false;


			set_size_for_layers(_layers_size);
			set_orientation_for_layers();
	   }

		node_autoencoder::~node_autoencoder(){}


		void node_autoencoder::loop()
		{
			CommandType command;
			do
			{
				//si aspettano eventuali nodi rimasti indietro
				MPI_Barrier(MPI_COMM_WORLD);

				//ottengo il comando (dipende dalla classe)
				command = wait_for_command();

				//getting times
				timeval wt0, wt1;
				oslog << "----  Executing command: " << std::to_string(static_cast<int>(command)) << " -------\n\n";
				gettimeofday(&wt0, NULL);

				switch(command){

					case CommandType::train:

						//1. Si apprendono le RBM per ciascun layer
						oslog << "Imparando le RBM...\n";
						oslog << "Numero di RBM da apprendere: " <<  number_of_rbm_to_learn <<"\n";
						oslog << "Numero di RBM gia apprese: " << trained_rbms << "\n";
						oslog << "Numero di layer finali: " <<  number_of_final_layers <<"\n";
						oslog.flush();

						//il numero del minibatch non può essere più grande del numero di esempi
						rbm_size_minibatch = std::min(rbm_size_minibatch, number_of_samples);

						train_rbm();

						if(!fine_tuning_finished)
							fine_tuning();

					break;

					case CommandType::load_pars:
						load_parameters();
						break;

					case CommandType::save_pars:
						save_parameters();
						break;

					case CommandType::delete_pars_file:
						std::remove(get_path_file().c_str());
						std::cout << "File deleted: " + get_path_file() + "\n";
						break;

					case CommandType::reconstruct_image:

						//reconstruct each sample
						for(uint i = 0; i != number_of_samples; i++)
							reconstruct();
						break;

					case CommandType::exit:
						break;
				}

				//printing executing times
				gettimeofday(&wt1, NULL);
				print_sec_gtd(oslog, wt0, wt1, mpi_rank);
				print_sec_gtd(std::cout, wt0, wt1, mpi_rank);
			}
			while(command != CommandType::exit);
		}


		//metodo base rimpiazzabile dal master
		CommandType node_autoencoder::wait_for_command()
		{
			CommandType command;

			//get command from master other node
			MPI_Bcast(&command,1, MPI_INT, 0, MPI_COMM_WORLD);

			if(command == CommandType::train || command == CommandType::reconstruct_image)
			{
				//ottengo numero di esempi
				MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);
			}
			else if (command == CommandType::load_pars || command == CommandType::save_pars)
			{
				//lettura cartella dei parametri
				char char_path_file[MAX_FOLDER_PARS_LENGTH];
				MPI_Bcast(char_path_file, MAX_FOLDER_PARS_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

				if(strcmp(char_path_file, ".") != 0)
					folder_parameters_path = string(char_path_file);
			}

			return command;
		}



		void node_autoencoder::set_size_for_layers(const my_vector<int>& _layers_size_source)
		{
			//_layers_size contiene la grandezza dei layer fino a quello centrale
			//in fase di rollup verranno creati altri layer per la ricostruzione
			layers_size = my_vector<int>(number_of_final_layers);
			for(uint i = 0; i !=_layers_size_source.size(); i++)
			{
			   layers_size[i] = _layers_size_source[i];

			   //si copia la grandezza del layer per il layer da ricostruire
			   uint rec_layer = number_of_final_layers - i - 1;
			   layers_size[rec_layer] = layers_size[i];
			}
		}

		void node_autoencoder::set_orientation_for_layers()
		{
			//si determina l'orientamento della griglia per ciascun layer
			//di default le righe prendono gli input e le colonne gli output
			orientation_grid = my_vector<GridOrientation>(number_of_final_layers - 1);
			for(uint i = 0; i != number_of_rbm_to_learn; i++)
			{
				int input_nodes = layers_size[i];
				int output_nodes = layers_size[i + 1];

				//c'è da considerare il fatto che l'orientamento è deciso in base al costo della comunicazione
				//di fatto significa se se i nodi di output sono maggiori di quelli di input, allora si inverte l'orientamento
				GridOrientation orientation = ((3 * input_nodes) < (4 * output_nodes)) ? GridOrientation::col_first : GridOrientation::row_first;

				orientation_grid[i] = orientation;

				//dato che il rollup si basa sul riutilizzo dei pesi, e dato che il
				//numero di nodi di input e di output è invertito tra il layer di ricostruzione e quello di encoding
				//l'orientamento sarà necessariamente inverso
				int rec_layer = number_of_final_layers - i - 2;
				orientation_grid[rec_layer] = orientation == GridOrientation::col_first ?
						GridOrientation::row_first : GridOrientation::col_first;
			}
		}
}

