/*
 * node_cell_autoencoder.h
 *
 *  Created on: 19 nov 2020
 *      Author: giovanni
 */

#ifndef NODE_CELL_AUTOENCODER_H_
#define NODE_CELL_AUTOENCODER_H_



#include "node_autoencoder.h"


namespace parallel_autoencoder{

	class node_cell_autoencoder : public node_autoencoder{

	private:

		vector<vector<vector<float>>> layers_weights;

		int row_number;
		int col_number;

		vector<MP_Comm_MasterSlave> accs_row_comm;
		vector<MP_Comm_MasterSlave> accs_col_comm;

		vector<vector<MP_Comm_MasterSlave>> acc_hid_comm_for_layer;
		vector<vector<MP_Comm_MasterSlave>> acc_vis_comm_for_layer;
	public:


		node_cell_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
				int _total_accumulators, int _grid_row, int _grid_col,
				std::ostream& _oslog, int _mpi_rank,
				int _row_number, int _col_number,
				vector<MP_Comm_MasterSlave>& _accs_row_comm, vector<MP_Comm_MasterSlave>& _accs_col_comm)
		: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
		{
			row_number = _row_number;
			col_number = _col_number;

			layers_weights = vector<vector<vector<float>>>(number_of_final_layers - 1);

			accs_row_comm = _accs_row_comm;
			accs_col_comm = _accs_col_comm;

			//si determinano quali comunicatori bisogna utilizzare ad ogni passo a seconda dell'orientamento scelto
			acc_hid_comm_for_layer = vector<vector<MP_Comm_MasterSlave>>(orientation_grid.size());
			acc_vis_comm_for_layer = vector<vector<MP_Comm_MasterSlave>>(orientation_grid.size());
			for(uint i = 0; i < orientation_grid.size(); i++)
			{
				const GridOrientation orientation_for_vis = orientation_grid[i];
				const GridOrientation orientation_for_hid = orientation_for_vis == row_first ? col_first : row_first;

				//calcolo del numero di nodi visibili o nascosti rappresentati per la cella corrente
				const int n_visible_units = layers_size[i];
				const int n_hidden_units = layers_size[i + 1];

				//b) che posizione rappresenta la cella per il vettore visibile/nascosto
				const auto my_vis_number = orientation_for_vis == row_first ? row_number : col_number;
				const auto my_hid_number = orientation_for_vis == row_first ? col_number : row_number;

				//comunicatori per nodi visibili e nascosti
				acc_vis_comm_for_layer[i] = orientation_for_vis == row_first ? accs_row_comm : accs_col_comm;
				acc_hid_comm_for_layer[i] = orientation_for_vis == row_first ? accs_col_comm : accs_row_comm;

				//calcolo delle size per comm acc verso le righe e verso le colonne
				calc_comm_sizes(orientation_for_vis, acc_vis_comm_for_layer[i] , false, my_vis_number, n_visible_units);
				calc_comm_sizes(orientation_for_hid, acc_hid_comm_for_layer[i], false, my_hid_number, n_hidden_units);
			}
		}

		CommandType wait_for_command()
		{
			CommandType command;

			//get command from master other node
			MPI_Bcast(&command,1, MPI_INT, 0, MPI_COMM_WORLD);

			if(command == train)
			{
				//ottengo numero di esempi
				MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);
			}
			else if (command == load_pars || command == save_pars)
			{
				//lettura cartella dei parametri
				char char_path_file[MAX_FOLDER_PARS_LENGTH];
				MPI_Bcast(char_path_file, MAX_FOLDER_PARS_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

				if(strcmp(char_path_file, ".") != 0)
					folder_parameters_path = string(char_path_file);
			}

			return command;
		}

		void SendVectorToReduce(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			/*MPI_Ireduce(vec.data(), MPI_IN_PLACE,
					comms[0].n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comms[0].comm , reqs );

			return ;*/



			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

			/*	MPI_Isend(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						0, 0, comm.comm , reqs + i);*/


				//std::cout << "displacement: " <<displacement << ", comm.n_items_to_send: "<<comm.n_items_to_send<<"\n";


				MPI_Ireduce(vec.data() + displacement, MPI_IN_PLACE,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm , reqs + i);


				displacement += comm.n_items_to_send;
			}
		}

		void ReceiveVector(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				MPI_Ibcast(vec.data() + displacement, comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm,  reqs + i);

				displacement += comm.n_items_to_send;
			}
		}
		
		
		void get_my_visible_hidden_units(const int layer_number, int& n_my_visible_units, int& n_my_hidden_units)
		{
			//orientamento grid
			const GridOrientation orientation_for_vis = orientation_grid[layer_number];

			//calcolo del numero di nodi visibili o nascosti rappresentati per la cella corrente
			const int n_visible_units = layers_size[layer_number];
			const int n_hidden_units = layers_size[layer_number + 1];

			//a) in quante parti deve essere diviso il vettore dei visibili/nascosti
			const auto total_for_vis = orientation_for_vis == row_first ? grid_rows : grid_cols;
			const auto total_for_hid = orientation_for_vis == row_first ? grid_cols : grid_rows;

			//b) che posizione rappresenta la cella per il vettore visibile/nascosto
			const auto my_vis_number = orientation_for_vis == row_first ? row_number : col_number;
			const auto my_hid_number = orientation_for_vis == row_first ? col_number : row_number;

			n_my_visible_units = get_units_for_node(n_visible_units, total_for_vis, my_vis_number);
			n_my_hidden_units = get_units_for_node(n_hidden_units, total_for_hid, my_hid_number);
		}



		void train_rbm()
		{
		    //1. Si apprendono le RBM per ciascun layer
			//se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
			for(int layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
			{
				//comunicatori per nodi visibili e nascosti
				auto& comms_for_vis = acc_vis_comm_for_layer[layer_number];
				auto& comms_for_hid = acc_hid_comm_for_layer[layer_number];

				int n_my_visible_units, n_my_hidden_units;
				get_my_visible_hidden_units(layer_number,n_my_visible_units, n_my_hidden_units);

				//RBM
				auto& weights = layers_weights[layer_number];

				//la matrice dei pesi per il layer in questione,
				//possiede grandezza VxH (unità visibili per unità nascoste)
				//si riserva lo spazio necessario
				weights = vector<vector<float>>(n_my_visible_units, vector<float>(n_my_hidden_units));

				//inizializzazione pesi
				for(auto& vec : weights)
					for(auto& v : vec)
						v = sample_gaussian_distribution(rbm_initial_weights_mean, rbm_initial_weights_variance, generator);


				//layers visibili e nascosti, ricostruiti e non
				vector<float> visible_units1(n_my_visible_units);
				vector<float> hidden_units1(n_my_hidden_units);
				vector<float> rec_visible_units1(n_my_visible_units);
				vector<float> rec_hidden_units1(n_my_hidden_units);

				vector<float> visible_units2(n_my_visible_units);
				vector<float> hidden_units2(n_my_hidden_units);
				vector<float> rec_visible_units2(n_my_visible_units);
				vector<float> rec_hidden_units2(n_my_hidden_units);

				//gradienti calcolati per pesi
				vector<vector<float>> diff_weights(n_my_visible_units, vector<float>(n_my_hidden_units, 0.0));

				//Puntatori delle request asincrone
				MPI_Request *reqsVis1 = new MPI_Request[comms_for_vis.size()];
				MPI_Request *reqsVis1Ricezione = new MPI_Request[comms_for_vis.size()];
				MPI_Request *reqsVis2 = new MPI_Request[comms_for_vis.size()];
				MPI_Request *reqsVis2Ricezione = new MPI_Request[comms_for_vis.size()];

				MPI_Request *reqsHid1 = new MPI_Request[comms_for_hid.size()];
				MPI_Request *reqsHid1Ricezione = new MPI_Request[comms_for_hid.size()];
				MPI_Request *reqsHid2 = new MPI_Request[comms_for_hid.size()];
				MPI_Request *reqsHid2Ricezione = new MPI_Request[comms_for_hid.size()];


				//si avvia il processo di apprendimento per diverse epoche
				long current_index_sample = 0;
				float current_learning_rate;


				// A1) Async Ricezione V 1
				ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

				for(int epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					current_learning_rate = GetRBMLearningRate(epoch, layer_number);

					while(current_index_sample < (epoch + 1) * number_of_samples)
					{
						current_index_sample+=2; //ogni ciclo while gestisce due esempi

						//CONTRASTIVE DIVERGENCE
						//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
						//E' necessario però analizzare due esempi per volta


						// INIZIO SECONDO METODO
						//

						// A1) Wait ricezione V 1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);


						// B1) Async ricezione V
						ReceiveVector(comms_for_vis, visible_units2, reqsVis2);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units1, vector<float>(n_my_hidden_units), hidden_units1);

						// B1) Wait ricezione V
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);





						// A2) Async invio H 1
						SendVectorToReduce(comms_for_hid, hidden_units1, reqsHid1);

						// A3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units1, reqsHid1Ricezione);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units2, vector<float>(n_my_hidden_units), hidden_units2);

						// A2) Wait invio H
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						// A3) Wait ricezione H
						MPI_Waitall(comms_for_hid.size(), reqsHid1Ricezione, MPI_STATUSES_IGNORE);




						// B2) Async invio H
						SendVectorToReduce(comms_for_hid, hidden_units2, reqsHid2);

						// B3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units2, reqsHid2Ricezione);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units1, vector<float>(n_my_visible_units), rec_visible_units1);

						// B2) Wait invio H
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						// B3) Wait ricezione H
						MPI_Waitall(comms_for_hid.size(), reqsHid2Ricezione, MPI_STATUSES_IGNORE);



						// A4) Async invio Vrec
						SendVectorToReduce(comms_for_vis, rec_visible_units1, reqsVis1);

						// A5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units1, reqsVis1Ricezione);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units2, vector<float>(n_my_visible_units), rec_visible_units2);

						// A4) Wait invio V rec
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

						// A5) Wait ricezione V rec
						MPI_Waitall(comms_for_vis.size(), reqsVis1Ricezione, MPI_STATUSES_IGNORE);



						// B4) Async invio V rec
						SendVectorToReduce(comms_for_vis, rec_visible_units2, reqsVis2);

						// B5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units2, reqsVis2Ricezione);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units1, vector<float>(n_my_hidden_units), rec_hidden_units1);

						// B4) Wait invio Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

						// B5) Wait ricezione Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2Ricezione, MPI_STATUSES_IGNORE);



						// A6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units1, reqsHid1);

						// A7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units1, reqsHid1Ricezione);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units2, vector<float>(n_my_hidden_units), rec_hidden_units2);

						// A6) Wait invio H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						// A7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1Ricezione, MPI_STATUSES_IGNORE);




						// B6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units2, reqsHid2);

						// B7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units2, reqsHid2Ricezione);

						//gradiente
						//si calcolano i differenziali dei pesi
						for(uint i = 0; i < visible_units1.size(); i++)
						{
							for(uint j = 0; j < hidden_units1.size(); j++){
								diff_weights.at(i).at(j) +=
										   visible_units1.at(i) * hidden_units1.at(j)  //fattore positivo
										 - rec_visible_units1.at(i) * rec_hidden_units1.at(j); //fattore negativo
							}
						}

						// B6) Wait invio H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						// B7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid2Ricezione, MPI_STATUSES_IGNORE);



						// A1) Async ricezione V
						const bool other_samples = current_index_sample != rbm_n_training_epocs * number_of_samples;
						if(other_samples)
						{
							ReceiveVector(comms_for_vis, visible_units1, reqsVis1);
						}

						//gradiente
						//si calcolano i differenziali dei pesi
						for(uint i = 0; i < visible_units2.size(); i++)
						{
							for(uint j = 0; j < hidden_units2.size(); j++){
								diff_weights.at(i).at(j) +=
										  visible_units2.at(i) * hidden_units2.at(j)  //fattore positivo
										 - rec_visible_units2.at(i) * rec_hidden_units2.at(j); //fattore negativo
							}
						}


						//FINE SECONDO METODO


/*

						// A1) Wait ricezione V 1
						std::cout << "[Cell" << row_number << "x" << col_number << ", wait ricezione V1];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);


						// B1) Async ricezione V
						ReceiveVector(comms_for_vis, visible_units2, reqsVis2);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units1, vector<float>(n_my_hidden_units), hidden_units1);


						// A2) Async invio H 1
						SendVectorToReduce(comms_for_hid, hidden_units1, reqsHid1);

						// B1) Wait ricezione V
						std::cout << "[Cell" << row_number << "x" << col_number << ", wait ricezione V2];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);


						// A3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units1, reqsHid1Ricezione);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units2, vector<float>(n_my_hidden_units), hidden_units2);


						// A2) Wait invio H
						 std::cout << "[Cell" << row_number << "x" << col_number << ",  wait invio H1];\n";
					    MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);


						// B2) Async invio H
						SendVectorToReduce(comms_for_hid, hidden_units2, reqsHid2);

						// A3) Wait ricezione H
						std::cout << "[Cell" << row_number << "x" << col_number << ", wait ricezione H1];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid1Ricezione, MPI_STATUSES_IGNORE);
						std::cout << "[Cell" << row_number << "x" << col_number << ", OK wait ric H1];\n";


						// B3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units2, reqsHid2Ricezione);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units1, vector<float>(n_my_visible_units), rec_visible_units1);


						// B2) Wait invio H
						std::cout << "[Cell" << row_number << "x" << col_number << ",  wait invio H2];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);


						// A4) Async invio Vrec
						SendVectorToReduce(comms_for_vis, rec_visible_units1, reqsVis1);

						// B3) Wait ricezione H
						std::cout << "[Cell" << row_number << "x" << col_number << ",  wait ricezione H2];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid2Ricezione, MPI_STATUSES_IGNORE);

						// A5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units1, reqsVis1Ricezione);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units2, vector<float>(n_my_visible_units), rec_visible_units2);


						// A4) Wait invio V rec
						std::cout << "[Cell" << row_number << "x" << col_number << ", wait invio Vrec1];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

						// B4) Async invio V rec
						SendVectorToReduce(comms_for_vis, rec_visible_units2, reqsVis2);

						// A5) Wait ricezione V rec
						std::cout << "[Cell" << row_number << "x" << col_number << ", wait ricezione Vrec1];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis1Ricezione, MPI_STATUSES_IGNORE);

						// B5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units2, reqsVis2Ricezione);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units1, vector<float>(n_my_hidden_units), rec_hidden_units1);


						// B4) Wait invio Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

						// A6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units1, reqsHid1);

						// B5) Wait ricezione Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2Ricezione, MPI_STATUSES_IGNORE);

						// A7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units1, reqsHid1Ricezione);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units2, vector<float>(n_my_hidden_units), rec_hidden_units2);


						// A6) Wait invio H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						// B6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units2, reqsHid2);

						// A7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1Ricezione, MPI_STATUSES_IGNORE);

						// B7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units2, reqsHid2Ricezione);

						//gradiente
						//si calcolano i differenziali dei pesi
						for(uint i = 0; i < visible_units1.size(); i++)
						{
							for(uint j = 0; j < hidden_units1.size(); j++){
								diff_weights.at(i).at(j) +=
										   visible_units1.at(i) * hidden_units1.at(j)  //fattore positivo
										 - rec_visible_units1.at(i) * rec_hidden_units1.at(j); //fattore negativo
							}
						}

						std::cout << "Cell Gradienti calcolati\n";

						// B6) Wait invio H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						// B7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid2Ricezione, MPI_STATUSES_IGNORE);

						// A1) Async ricezione V
						const bool other_samples = current_index_sample != rbm_n_training_epocs * number_of_samples;
						if(other_samples)
						{
							ReceiveVector(comms_for_vis, visible_units1, reqsVis1);
						}

						//gradiente
						//si calcolano i differenziali dei pesi
						for(uint i = 0; i < visible_units2.size(); i++)
						{
							for(uint j = 0; j < hidden_units2.size(); j++){
								diff_weights.at(i).at(j) +=
										  visible_units2.at(i) * hidden_units2.at(j)  //fattore positivo
										 - rec_visible_units2.at(i) * rec_hidden_units2.at(j); //fattore negativo
							}
						}
*/

						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0){

							update_parameters(current_learning_rate, weights, diff_weights,rbm_size_minibatch);
						}

					} //fine esempio
				} //fine epoca


				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0){

					//modifica pesi
					update_parameters(current_learning_rate, weights, diff_weights, n_last_samples);
				}



				//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)

				// 1) Async ricezione V
				ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

				for(current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					// 1) Wait ricezione V
					MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

					visible_units2 = visible_units1; //copia valori in un altro buffer

					// 1) Async ricezione V
					if(current_index_sample != number_of_samples - 1)
						ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

					//Calcolo
					matrix_transpose_vector_multiplication(weights, visible_units2, vector<float>(n_my_hidden_units), hidden_units1);

					// 3) Wait Invio H
					if(current_index_sample > 0)
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

					// 3) Async Invio H
					hidden_units2 = hidden_units1; //utilizzo un altro buffer
					SendVectorToReduce(comms_for_hid, hidden_units1, reqsHid1);
				}

				// 3) Wait Invio H
				MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);


				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();


			}

		}

		//dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
		 //la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
	    inline void update_parameters(
	    		const float current_learning_rate,
		        vector<vector<float>> &weights,
		        vector<vector<float>> &diff_weights,
		        const int number_of_samples
		        ){
			    //todo il numero lo si dovrebbe già avere
				 const int n_j = weights[0].size();

				//si precalcola il fattore moltiplicativo
				//dovendo fare una media bisogna dividere per il numero di esempi
				const float mult_factor = current_learning_rate / number_of_samples;

				//diff per pesi e bias visibili
				for(uint i = 0; i < weights.size(); i++)
				{
					for(int j = 0; j < n_j; j++){
					   weights.at(i).at(j) += diff_weights.at(i).at(j) * mult_factor;

					   diff_weights.at(i).at(j) = diff_weights.at(i).at(j) * rbm_momentum;//inizializzazione per il momentum
					}
				}
		}



	    void get_activation_output_layers(vector<vector<float>>& activation_layers,vector<vector<float>>& output_layers)
	    {
			for(uint l = 0; l < activation_layers.size(); l++)
			{
				int n_my_visible_units, n_my_hidden_units;
				get_my_visible_hidden_units(l,n_my_visible_units, n_my_hidden_units);
				
				activation_layers.at(l) = vector<float>(n_my_visible_units);
				output_layers.at(l) = vector<float>(n_my_hidden_units);
			}
	    }

	    void forward_pass(vector<vector<float>>& activation_layers,vector<vector<float>>& output_layers)
	    {
	    	//1. forward pass
			for(int l = 0; l < number_of_final_layers - 1; l++){

				//comunicatori
				auto& comms_for_vis = acc_vis_comm_for_layer[l];
				auto& comms_for_hid = acc_hid_comm_for_layer[l];

				//bias, input e output vectors
				auto& weights = layers_weights.at(l);
				auto& input = activation_layers.at(l);
				auto& output = output_layers.at(l);


				//Ricezione da accumulatori
				MPI_Request vis_requests[comms_for_vis.size()];
				ReceiveVector(comms_for_vis, input, vis_requests);
				MPI_Waitall(comms_for_vis.size(), vis_requests, MPI_STATUSES_IGNORE);

				//Calcolo matriciale H = V * W
				matrix_transpose_vector_multiplication(weights, input, vector<float>(output.size()), output);

				//Invio ad accumulatori
				MPI_Request reqs_hid[comms_for_hid.size()];
				SendVectorToReduce(comms_for_hid, output, reqs_hid);

			} //fine forward
	    }


		void fine_tuning()
		{
			//Roll-up
			//una volta appresi i pesi, bisogna creare una rete di tipo feed forward
			//la rete feed forward dell'autoencoder possiede il doppio dei layer hidden,
			//ad eccezione del layer più piccolo che di fatto serve a memorizzare l'informazione in maniera più corta
			for(int trained_layer = number_of_rbm_to_learn - 1; trained_layer >= 0;  trained_layer--)
			{
				int new_layer = number_of_final_layers - trained_layer - 1;

				//memorizzo i pesi trasposti nel layer feed forward, attenzione agli indici
				int index_weights_dest = new_layer - 1;

				//si salva la trasposta dei pesi
				auto& layer_weights_source = layers_weights.at(trained_layer);
				auto& layer_weights_dest = layers_weights.at(index_weights_dest);

				layer_weights_dest = vector<vector<float>>(layer_weights_source[0].size(), vector<float>(layer_weights_source.size()));

				//todo assicurarsi che a livello globale la matrice dei pesi sia stata trasposta
				//grazie all'orientamento inverso dovrebbe essere così
				matrix_transpose(layer_weights_source, layer_weights_dest);
			}


			//INIZIO FINE TUNING
			//si riserva lo spazio necessario per l'attivazione di ogni layer
			// e per l'output restituito ad ogni layer
			vector<vector<float>> activation_layers(orientation_grid.size());
			vector<vector<float>> output_layers(orientation_grid.size());
			get_activation_output_layers(activation_layers, output_layers);

			//per ogni epoca...
			for(int epoch = 0; epoch < fine_tuning_n_training_epocs; epoch++)
			{
				for(int current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					//1. forward pass
					forward_pass(activation_layers, output_layers);

					//2. backward pass
					//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
					for(int l = number_of_final_layers - 1; l >= 1; l--){

						//comunicatori
						auto& comms_for_vis = acc_vis_comm_for_layer[l - 1];
						auto& comms_for_hid = acc_hid_comm_for_layer[l - 1];

						//si aggiornano i pesi tra l'output e l'input layer
						auto& weights_to_update = layers_weights.at(l - 1);

						auto& output_layer = output_layers.at(l - 1);
						auto& input_layer = activation_layers.at(l - 1);

						//vettori delta in input e calcolati
						auto deltas_to_accs = vector<float>(input_layer.size());
						auto deltas_from_accs = vector<float>(output_layer.size());

						//check misure
						assert(output_layer.size() == weights_to_update.at(0).size());
						assert(input_layer.size() == weights_to_update.size());

						//ricevi delta da accumulatori
						MPI_Request req_from_acc[comms_for_hid.size()];
						ReceiveVector(comms_for_hid, deltas_from_accs, req_from_acc);
						MPI_Waitall(comms_for_hid.size(), req_from_acc, MPI_STATUSES_IGNORE);

						//calcola delta pesati (li si moltiplicano per la matrice dei pesi)
						matrix_vector_multiplication(weights_to_update, deltas_from_accs, vector<float>(deltas_to_accs.size()), deltas_to_accs);

						//Invia delta pesati agli accumulatori (all'ultimo passo non è necessario)
						if(l > 1)
						{
							MPI_Request reqs_vis[comms_for_vis.size()];
							SendVectorToReduce(comms_for_vis, deltas_to_accs, reqs_vis);
						}

						//applico gradiente per la matrice dei pesi
						for(uint i = 0; i < input_layer.size(); i++){
							for(uint j = 0; j < output_layer.size(); j++){
								//delta rule
								weights_to_update.at(i).at(j) +=
										fine_tuning_learning_rate
										* deltas_from_accs.at(j)
										* input_layer.at(i);
							}
						}
					}

				} //fine esempio
			} //fine epoca

			//allenamento concluso
			fine_tuning_finished = true;
			save_parameters();


			//si aspettano eventuali nodi rimasti indietro
			MPI_Barrier(MPI_COMM_WORLD);
		}


	    vector<float> reconstruct(){


	    	vector<vector<float>> activation_layers(orientation_grid.size());
			vector<vector<float>> output_layers(orientation_grid.size());
			get_activation_output_layers(activation_layers, output_layers);

	    	//1. forward pass
			forward_pass(activation_layers, output_layers);

			return output_layers[output_layers.size() - 1]; //dummy
	    }


	    string get_path_file(){

	    	return folder_parameters_path + "paral_cell_"+ std::to_string(row_number) + "x" + std::to_string(col_number) + ".txt";
	    }


		void save_parameters(){

			string path_file = get_path_file();

			// Create an input filestream
			std::ofstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

			//salvataggio di pesi, bias
			int layer_number;
			for(layer_number = 0; layer_number < trained_rbms; layer_number++)
			{
				auto& weights = layers_weights[layer_number];

				myFile << "_rbm_" << weights.size() << "x" << weights[0].size() << "__,";
				for(auto& row : weights)
					for(auto& v : row)
						myFile << fixed << setprecision(F_PREC) << v << ",";
				myFile << endl;
			}

			if(fine_tuning_finished)
			{
				while(layer_number < number_of_final_layers - 1)
				{
					auto& weights = layers_weights[layer_number];

					myFile << "_rec_" << weights.size() << "x" << weights[0].size() << "__,";
					for(auto& row : weights)
						for(auto& v : row)
							myFile << fixed << setprecision(F_PREC) << v << ",";
					myFile << endl;

					layer_number++;
				}
			}

			myFile.close();

		}

	    void load_parameters(){


	    	string path_file = get_path_file();

			fine_tuning_finished = false;
			trained_rbms = 0;

			// Create an input filestream
			std::ifstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

			// Helper vars
			std::string line;

			//variabili che fanno riferimento al layer nel quale si salveranno i parametri
			int n_my_visible_units;
			int n_my_hidden_units;

			vector<vector<float>> *current_weights;

			//Si leggono le linee che contengono i pesi delle rbm apprese
			bool other_lines = false;
			while(std::getline(myFile, line))
			{
				//se abbiamo letto tutti i pesi delle rbm si esce da questo ciclo
				if(trained_rbms == number_of_rbm_to_learn){
					//se ci sono altre linee, vuol dire che si possiedono i parametri dei layer di ricostruzione
					other_lines = true;
					break;
				}
				
				//si ottengono grandezze
				get_my_visible_hidden_units(trained_rbms, n_my_visible_units, n_my_hidden_units);

				//matrice pesi
				layers_weights.at(trained_rbms) = vector<vector<float>>(n_my_visible_units, vector<float>(n_my_hidden_units));
				current_weights = &layers_weights.at(trained_rbms);

				//questa variabile memorizza il numero di rbm apprese
				trained_rbms++;


				// Create a stringstream of the current line
				std::stringstream ss(line);

				//riga dei pesi
				ss.ignore(100, ',');

				for(int i = 0; i < n_my_visible_units; i++)
					for(int j = 0; j < n_my_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_weights->at(i).at(j);
					}
			}


			//se ci sono altre linee da analizzare vuol dire che si aggiornano i pesi dei layer di ricostruzione
			if(other_lines)
			{
				//indice del layer contenente pesi o bias
				int current_layer = (number_of_final_layers - 1) / 2;
				do
				{
					//grandezze
					get_my_visible_hidden_units(current_layer, n_my_visible_units, n_my_hidden_units);

					//matrice pesi
					layers_weights.at(current_layer) = vector<vector<float>>(n_my_visible_units, vector<float>(n_my_hidden_units));
					current_weights = &layers_weights.at(current_layer);

					current_layer++;

					// Create a stringstream of the current line
					std::stringstream ss(line);

					//riga dei pesi
					ss.ignore(100, ',');
					for(int i = 0; i < n_my_visible_units; i++)
						for(int j = 0; j < n_my_hidden_units; j++){
						   if(ss.peek() == ',') ss.ignore();
						   ss >>  current_weights->at(i).at(j);
						}

				}
				while(std::getline(myFile, line));

				//il training si considera concluso
				fine_tuning_finished = true;
			}

			// Close file
			myFile.close();
	    }
	};

}


#endif /* NODE_CELL_AUTOENCODER_H_ */
