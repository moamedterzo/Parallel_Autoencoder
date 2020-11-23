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
	public:


		node_cell_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
				int _total_accumulators, int _grid_row, int _grid_col,
				std::ostream& _oslog, int _mpi_rank,
				int _row_number, int _col_number,
				vector<MP_Comm_MasterSlave>_accs_row_comm, vector<MP_Comm_MasterSlave> _accs_col_comm)
		: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
		{
			row_number = _row_number;
			col_number = _col_number;

			layers_weights = vector<vector<vector<float>>>(number_of_final_layers - 1);

			accs_row_comm = _accs_row_comm;
			accs_col_comm = _accs_col_comm;
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

			return command;
		}

		void SendVectorToReduce(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				MPI_Ireduce(vec.data() + displacement, MPI_IN_PLACE,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						comm.root_id, comm.comm , reqs + i);

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

				MPI_Irecv(vec.data() + displacement, comm.n_items_to_send, mpi_datatype_tosend,
						 0, comm.root_id, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}



		void train_rbm()
		{
		    //1. Si apprendono le RBM per ciascun layer
			//se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
			for(int layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
			{
				//orientamento grid
				const GridOrientation orientation_for_vis = orientation_grid[layer_number];
				const GridOrientation orientation_for_hid = orientation_for_vis == row_first ? col_first : row_first;

				//comunicatori per nodi visibili e nascosti
				auto& comms_for_vis = orientation_for_vis == row_first ? accs_row_comm : accs_col_comm;
				auto& comms_for_hid = orientation_for_hid == row_first ? accs_row_comm : accs_col_comm;

				int n_my_visible_units, n_my_hidden_units;

				{
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

					//calcolo delle size per comm acc verso le righe e verso le colonne
					calc_comm_sizes(orientation_for_vis, comms_for_vis, false, my_vis_number, n_visible_units);
					calc_comm_sizes(orientation_for_hid, comms_for_hid, false, my_hid_number, n_hidden_units);
				}

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
				MPI_Request *reqsVis2 = new MPI_Request[comms_for_vis.size()];

				MPI_Request *reqsHid1 = new MPI_Request[comms_for_hid.size()];
				MPI_Request *reqsHid2 = new MPI_Request[comms_for_hid.size()];


				//si avvia il processo di apprendimento per diverse epoche
				long current_index_sample = 0;


				// A1) Async Ricezione V 1
				ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

				for(int epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					//todo implementare per bene modifiche al learning rate in base all'epoca

					while(current_index_sample < (epoch + 1) * number_of_samples)
					{
						current_index_sample+=2; //ogni ciclo while gestisce due esempi

						//CONTRASTIVE DIVERGENCE
						//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
						//E' necessario però analizzare due esempi per volta

						// A1) Wait ricezione V 1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B1) Async ricezione V
						ReceiveVector(comms_for_vis, visible_units2, reqsVis2);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units1, vector<float>(n_my_hidden_units), hidden_units1);


						// A2) Async invio H 1
						SendVectorToReduce(comms_for_hid, hidden_units1, reqsHid1);

						// B1) Wait ricezione V
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units1, reqsHid1);

						//Prodotto matriciale V * W
						matrix_transpose_vector_multiplication(weights, visible_units2, vector<float>(n_my_hidden_units), hidden_units2);


						// A2) Wait invio H
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B2) Async invio H
						SendVectorToReduce(comms_for_hid, hidden_units2, reqsHid2);

						// A3) Wait ricezione H
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B3) Async ricezione H
						ReceiveVector(comms_for_hid, hidden_units2, reqsHid2);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units1, vector<float>(n_my_visible_units), rec_visible_units1);


						// B2) Wait invio H
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						// A4) Async invio Vrec
						SendVectorToReduce(comms_for_vis, rec_visible_units1, reqsVis1);

						// B3) Wait ricezione H
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						// A5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units1, reqsVis1);

						//Prodotto matriciale H * W
						matrix_vector_multiplication(weights, hidden_units2, vector<float>(n_my_visible_units), rec_visible_units2);


						// A4) Wait invio V rec
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B4) Async invio V rec
						SendVectorToReduce(comms_for_vis, rec_visible_units2, reqsVis2);

						// A5) Wait ricezione V rec
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B5) Async ricezione Vrec
						ReceiveVector(comms_for_vis, rec_visible_units2, reqsVis2);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units1, vector<float>(n_my_hidden_units), rec_hidden_units1);


						// B4) Wait invio Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units1, reqsHid1);

						// B5) Wait ricezione Vrec
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units1, reqsHid1);

						//Prodotto matriciale V' * W
						matrix_transpose_vector_multiplication(weights, rec_visible_units2, vector<float>(n_my_hidden_units), rec_hidden_units2);


						// A6) Wait invio H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B6) Async invio H rec
						SendVectorToReduce(comms_for_hid, rec_hidden_units2, reqsHid2);

						// A7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// A7) Async ricezione H rec
						ReceiveVector(comms_for_hid, rec_hidden_units2, reqsHid2);

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
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						// B7) Wait ricezione H rec
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

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


						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0){

							update_parameters(weights,	diff_weights,rbm_size_minibatch);

						}
					}
				}

				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0){

					//modifica pesi
					update_parameters(weights,	diff_weights, n_last_samples);
				}



				//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)

				// 1) Async ricezione V
				ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

				for(current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					// 1) Wait ricezione V
					MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);
					visible_units2 = visible_units1; //copia valori in un altro buffer

					// 1) Async ricezione V
					if(current_index_sample != number_of_samples - 1)
						ReceiveVector(comms_for_vis, visible_units1, reqsVis1);

					//Calcolo
					matrix_transpose_vector_multiplication(weights, visible_units2, vector<float>(n_my_hidden_units), hidden_units1);

					// 3) Wait Invio H
					MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

					// 3) Async Invio H
					hidden_units2 = hidden_units1; //utilizzo un altro buffer
					SendVectorToReduce(comms_for_hid, hidden_units1, reqsHid1);
				}

				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();
			}

		}

		//dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
		 //la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
	      inline void update_parameters(
		        vector<vector<float>> &weights,
		        vector<vector<float>> &diff_weights,
		        const int number_of_samples
		        ){
			    //todo il numero lo si dovrebbe già avere
				 const int n_j = weights[0].size();

				//si precalcola il fattore moltiplicativo
				//dovendo fare una media bisogna dividere per il numero di esempi
				const int mult_factor = rbm_learning_rate / number_of_samples;

				//diff per pesi e bias visibili
				for(uint i = 0; i < weights.size(); i++)
				{
					for(int j = 0; j < n_j; j++){
					   weights.at(i).at(j) += diff_weights.at(i).at(j) * mult_factor;

					   diff_weights.at(i).at(j) = diff_weights.at(i).at(j) * rbm_momentum / mult_factor;//inizializzazione per il momentum
					}
				}
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


			//todo resto
		}


	    vector<float> reconstruct(){}
		void save_parameters(){}
	    void load_parameters(){}
	};

}


#endif /* NODE_CELL_AUTOENCODER_H_ */
