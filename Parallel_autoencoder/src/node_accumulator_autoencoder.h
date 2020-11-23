/*
 * node_accumulator_autoencoder.h
 *
 *  Created on: 19 nov 2020
 *      Author: giovanni
 */

#ifndef NODE_ACCUMULATOR_AUTOENCODER_H_
#define NODE_ACCUMULATOR_AUTOENCODER_H_


#include "node_autoencoder.h"


#include <cassert>


namespace parallel_autoencoder{

	class node_accumulator_autoencoder : public node_autoencoder{

	private:
		vector<vector<float>> layer_biases;

		int k_number;

		//comunicatore da master a nodi accumulatori
		MP_Comm_MasterSlave master_accs_comm;
		vector<MP_Comm_MasterSlave> acc_rows_comm;
		vector<MP_Comm_MasterSlave> acc_cols_comm;
	public:

		node_accumulator_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
				int _total_accumulators, int _grid_row, int _grid_col,
				std::ostream& _oslog, int _mpi_rank,
				int _k_number,
				MP_Comm_MasterSlave _master_accs_comm, vector<MP_Comm_MasterSlave>_acc_rows_comm, vector<MP_Comm_MasterSlave> _acc_cols_comm)
		: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
		{
			k_number = _k_number;

			layer_biases = vector<vector<float>>(number_of_final_layers - 1); //tutti i layer hanno il bias tranne quello di input

			master_accs_comm = _master_accs_comm;
			acc_rows_comm = _acc_rows_comm;
			acc_cols_comm = _acc_cols_comm;
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


		void BroadcastVector(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			//Si invia il vettore alle righe/colonne di riferimento, per ognuna si usa il broadcast
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				MPI_Ibcast(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						comm.root_id, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void AccumulateVector(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				MPI_Ireduce(MPI_IN_PLACE, vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						comm.root_id, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void ReceiveFromMaster(vector<float>& vis_vec, MPI_Request *reqVis)
		{
			MPI_Iscatter(nullptr, 0, NULL,
					vis_vec.data(), vis_vec.size(), mpi_datatype_tosend,
					master_accs_comm.root_id, master_accs_comm.comm, reqVis);
		}



		void train_rbm()
		{
			//1. Si apprendono le RBM per ciascun layer
			//se sono state già apprese delle rbm, si passa direttamente alla prima da imparare
			for(int layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
			{
				const bool first_layer = layer_number == 0;
				const int index_reverse_layer = number_of_final_layers - layer_number - 2;

				//orientamento grid
				const GridOrientation orientation_for_vis = orientation_grid[layer_number];
				const GridOrientation orientation_for_hid = orientation_for_vis == row_first ? col_first : row_first;

				//comunicatori per nodi visibili e nascosti
				auto& comms_for_vis = orientation_for_vis == row_first ? acc_rows_comm : acc_cols_comm;
				auto& comms_for_hid = orientation_for_hid == row_first ? acc_rows_comm : acc_cols_comm;

				int n_my_visible_units, n_my_hidden_units;

				{
					//calcolo del numero di nodi visibili o nascosti rappresentati per l'accumulatore corrente
					const int n_visible_units = layers_size[layer_number];
					const int n_hidden_units = layers_size[layer_number + 1];

					n_my_visible_units = get_units_for_node(n_visible_units, total_accumulators, k_number);
					n_my_hidden_units = get_units_for_node(n_hidden_units, total_accumulators, k_number);

					//calcolo delle size per comm acc verso le righe e verso le colonne
					calc_comm_sizes(orientation_for_vis, comms_for_vis, true, k_number, n_visible_units);
					calc_comm_sizes(orientation_for_hid, comms_for_hid, true, k_number, n_hidden_units);
				}

				//RBM
				//inizializzazione bias
				auto& hidden_biases = layer_biases[layer_number];
				auto& visible_biases =  layer_biases[index_reverse_layer];

				visible_biases = vector<float>(n_my_visible_units, rbm_initial_biases_value);
				hidden_biases = vector<float>(n_my_hidden_units, rbm_initial_biases_value);

				//layers visibili e nascosti, ricostruiti e non
				vector<float> visible_units1(n_my_visible_units);
				vector<float> hidden_units1(n_my_hidden_units);
				vector<float> rec_visible_units1(n_my_visible_units);
				vector<float> rec_hidden_units1(n_my_hidden_units);

				vector<float> visible_units2(n_my_visible_units);
				vector<float> hidden_units2(n_my_hidden_units);
				vector<float> rec_visible_units2(n_my_visible_units);
				vector<float> rec_hidden_units2(n_my_hidden_units);


				//gradienti calcolati per pesi e bias
				vector<float> diff_visible_biases(n_my_visible_units, 0.0);
				vector<float> diff_hidden_biases(n_my_hidden_units, 0.0);

				//Puntatori delle request asincrone
				MPI_Request *reqsVis1 = new MPI_Request[comms_for_vis.size()];
				MPI_Request *reqsVis2 = new MPI_Request[comms_for_vis.size()];

				MPI_Request *reqsHid1 = new MPI_Request[comms_for_hid.size()];
				MPI_Request *reqsHid2 = new MPI_Request[comms_for_hid.size()];


				//si avvia il processo di apprendimento per diverse epoche
				long current_index_sample = 0;

				// A0) Sync ricevi input V 1 da nodo master
				ReceiveFromMaster(visible_units1, reqsVis1);
				MPI_Wait(reqsVis1, nullptr);

				// A1) Async Invio V 1
				BroadcastVector(comms_for_vis, visible_units1, reqsVis1);

				// B0) Async Ricevo secondo V (utilizzo il request dei visibili)
				ReceiveFromMaster(visible_units2, reqsVis2);

				for(int epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					//todo implementare per bene modifiche al learning rate in base all'epoca

					while(current_index_sample < (epoch + 1) * number_of_samples)
					{
						current_index_sample+=2; //ogni ciclo while gestisce due esempi

						//CONTRASTIVE DIVERGENCE
						//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
						//E' necessario però analizzare due esempi per volta

						// A2) Async ricevo H 1
						AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);

						// A1) Wait Invio V 1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B0) Wait ricezione input V 2 da master
						MPI_Wait(reqsVis2, nullptr);

						// B1) Async invio V 2
						BroadcastVector(comms_for_vis, visible_units2, reqsVis2);

						// A2) Wait ricezione H 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B2) Async ricevo H 2
						AccumulateVector(comms_for_hid, hidden_units2, reqsHid2);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < hidden_units1.size(); i++)
							hidden_units1.at(i) = sample_sigmoid_function(hidden_units1.at(i), generator);

						// B1) Wait Invio V 2
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A3) Async Invio H 1
						BroadcastVector(comms_for_hid, hidden_units1, reqsHid1);

						// B2) Wait ricezione H 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						// A4) Async ricezione Vrec 1
						AccumulateVector(comms_for_vis, rec_visible_units1, reqsVis1);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < hidden_units2.size(); i++)
							hidden_units2.at(i) = sample_sigmoid_function(hidden_units2.at(i), generator);

						// A3) Wait invio H 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B3) Async Invio H 2
						BroadcastVector(comms_for_hid, hidden_units2, reqsHid2);

						// A4) Wait ricezione Vrec 1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B4) Async ricezione Vrec 2
						AccumulateVector(comms_for_vis, rec_visible_units2, reqsVis2);

						//funzione Vrec 1
						//non si applica il campionamento
						if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
							for(uint i = 0; i < rec_visible_units1.size(); i++)
								rec_visible_units1.at(i) = sample_gaussian_distribution(rec_visible_units1.at(i), generator);
						else
							for(uint i = 0; i < rec_visible_units1.size(); i++)
								rec_visible_units1.at(i) = sigmoid(rec_visible_units1.at(i));


						// B3) Wait invio H 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						// A5) Async invio  Vrec 1
						BroadcastVector(comms_for_vis, rec_visible_units1, reqsVis1);

						// B4) Wait ricezione V rec 2
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A6) Async ricezione H rec 1
						AccumulateVector(comms_for_hid, rec_hidden_units1, reqsHid1);

						//funzione Vrec 2
						//non si applica il campionamento
						if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
							for(uint i = 0; i < rec_visible_units2.size(); i++)
								rec_visible_units2.at(i) = sample_gaussian_distribution(rec_visible_units2.at(i), generator);
						else
							for(uint i = 0; i < rec_visible_units2.size(); i++)
								rec_visible_units2.at(i) = sigmoid(rec_visible_units2.at(i));


						// A5) Wait invio V rec 1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, nullptr);

						// B5) Async invio V rec 2
						BroadcastVector(comms_for_vis, rec_visible_units2, reqsVis2);

						// A6) Wait ricezione H rec 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B6) Async ricezione H rec 2
						AccumulateVector(comms_for_hid, rec_hidden_units2, reqsHid2);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < rec_hidden_units1.size(); i++)
							   rec_hidden_units1.at(i) = sigmoid(rec_hidden_units1.at(i));

						// B5) Wait invio V rec 2
						MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

						// A7) Async invio H rec 1
						BroadcastVector(comms_for_hid, rec_hidden_units1, reqsHid1);

						// B6) Wait ricezione H rec 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < rec_hidden_units2.size(); i++)
							rec_hidden_units2.at(i) = sigmoid(rec_hidden_units2.at(i));

						// A7) Wait invio H rec 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

						// B7) Async invio H rec 2
						BroadcastVector(comms_for_hid, rec_hidden_units2, reqsHid2);


						// A0) Async ricezione nuovo input V1 da master (sempre se ci sono ancora esempi)
						const bool other_samples = current_index_sample != rbm_n_training_epocs * number_of_samples;
						if(other_samples)
						{
							ReceiveFromMaster(visible_units1, reqsVis1);
						}

						//gradiente 1
						//si calcolano i differenziali
						//dei pesi e bias visibili
						for(uint i = 0; i < visible_units1.size(); i++)
							diff_visible_biases.at(i) += visible_units1.at(i) - rec_visible_units1.at(i);

						//dei bias nascosti
						for(uint j = 0; j < hidden_units1.size(); j++)
							diff_hidden_biases.at(j) += hidden_units1.at(j) - rec_hidden_units1.at(j);


						// B7) Wait invio H rec 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, nullptr);

						if(other_samples)
						{
							// A0) Wait ricezione nuovo input V1
							MPI_Wait(reqsVis1, nullptr);

							// A1) Async Invio V 1
							BroadcastVector(comms_for_vis, visible_units1, reqsVis1);

							// B0) Async Ricevo secondo V (utilizzo il request dei visibili)
							ReceiveFromMaster(visible_units2, reqsVis2);
						}


						//gradiente 2
						//si calcolano i differenziali
						//dei pesi e bias visibili
						for(uint i = 0; i < visible_units2.size(); i++)
							diff_visible_biases.at(i) += visible_units2.at(i) - rec_visible_units2.at(i);

						//dei bias nascosti
						for(uint j = 0; j < hidden_units2.size(); j++)
							diff_hidden_biases.at(j) += hidden_units2.at(j) - rec_hidden_units2.at(j);



						//applicazione gradienti
						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0){

							update_parameters(hidden_biases, visible_biases,
									diff_visible_biases, diff_hidden_biases, rbm_size_minibatch);

						}
					}
				}

				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0){

					//modifica pesi
					update_parameters(hidden_biases, visible_biases,
								diff_visible_biases, diff_hidden_biases, n_last_samples);
				}


				//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)

				//0) Async ricevi input V da nodo master
				ReceiveFromMaster(visible_units1, reqsVis1);

				for(current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					// 0) Attendo ricezione vettore Visibile
					MPI_Wait(reqsVis1, nullptr);

					//1) Async Invio V (si utilizza un altro buffer)
					visible_units2 = visible_units1;
					BroadcastVector(comms_for_vis, visible_units2, reqsVis2);

					//2) Async ricevo H
					AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);

					//1) Wait Invio V
					MPI_Waitall(comms_for_vis.size(), reqsVis2, nullptr);

					//2) Wait Ricezione H
					MPI_Waitall(comms_for_hid.size(), reqsHid1, nullptr);

					//0) Async ricevi input V da nodo master
					if(current_index_sample != number_of_samples - 1)
						ReceiveFromMaster(visible_units1, reqsVis1);

					//funzione sigmoide su H
					for(uint i = 0; i < hidden_units1.size(); i++)
						hidden_units1.at(i) = sigmoid(hidden_units1.at(i));

					// 4) Wait invio H a master
					MPI_Wait(reqsHid2, nullptr);

					// 4) Invio async H a master (un altro buffer)
					hidden_units2 = hidden_units1;
					MPI_Igather(hidden_units2.data(), hidden_units2.size(), mpi_datatype_tosend,
							nullptr, 0, NULL,
							master_accs_comm.root_id, master_accs_comm.comm, reqsHid2);
				}

				// 4) Wait invio H a master
				MPI_Wait(reqsHid2, nullptr);

				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();
			}

		}







		void fine_tuning()
		{
			//Rollup per i bias già effettuato in fase di apprendimento delle rbm


			//si riserva lo spazio necessario per l'attivazione di ogni layer
			//e per i vettori che conterranno i valori delta per la back propagation
			vector<vector<float>> activation_layers(number_of_final_layers);
			for(int l = 0; l < activation_layers.size(); l++)
				activation_layers.at(l) = vector<float>(layers_size.at(l)); //la grandezza è memorizzata nel vettore layers_size

			//si esclude il primo layer dato che non possiede pesi da aggiornare
			vector<vector<float>> delta_layers(number_of_final_layers - 1);
			for(int l = 0; l < delta_layers.size(); l++)
				delta_layers.at(l) = vector<float>(layers_size.at(l + 1)); //la grandezza è memorizzata nel vettore layers_size

			int central_layer = number_of_final_layers / 2 - 1;

			//per ogni epoca...
			for(int epoch = 0; epoch < fine_tuning_n_training_epocs; epoch++)
			{

				//per ciascun esempio...
				while(true){


					//1. forward pass
					for(int l = 1; l < number_of_final_layers; l++){

						auto& biases = layer_biases.at(l - 1);
						auto& input = activation_layers.at(l - 1);
						auto& activation_layer = activation_layers.at(l);

						//todo calcolo

						//si applica la funzione sigmoide
						//se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
						//per ottenere un valore binario
						if(l == central_layer)
							for(auto& v : activation_layer)
								v = round(sigmoid(v));
						else
							for(auto& v : activation_layer)
								v = sigmoid(v);
					}


					//2. backward pass
					//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
					vector<float> output_deltas;
					vector<float> current_deltas;

					for(int l = number_of_final_layers - 1; l >= 1; l--){

						//si aggiornano i pesi tra l'output e l'input layer
						auto& biases_to_update = layer_biases.at(l - 1);

						auto& output_layer = activation_layers.at(l);
						auto& input_layer = activation_layers.at(l - 1);

						//check
						assert(output_layer.size() == biases_to_update.size());

						//si calcoleranno i delta per il layer corrente
						if(l == number_of_final_layers - 1)
						{
							//layer di output
							current_deltas = vector<float>(output_layer.size(), 0.0);

							auto& first_activation_layer = activation_layers.at(0);

							//calcolo dei delta per il layer di output
							// delta = y_i * (1 - y_i) * reconstruction_error
							for(int j = 0; j < output_layer.size(); j++)
							{
							   current_deltas[j] = output_layer.at(j)
									   * (1 - output_layer.at(j))
									   * (first_activation_layer.at(j) - output_layer.at(j));
							}
						}
						else
						{
							//layer nascosto

							//si memorizzano i delta del passo precedente
							output_deltas = vector<float>(current_deltas);
							current_deltas = vector<float>(output_layer.size(), 0.0);

						}

						//seguendo la delta rule, si applica il gradiente anche i bias
						for(int j = 0; j < biases_to_update.size(); j++){
							biases_to_update.at(j) +=
										fine_tuning_learning_rate
										* current_deltas.at(j);
						}
					}
				}
			}

			//allenamento concluso
			fine_tuning_finished = true;
			save_parameters();

		}


		//dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
		//la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
		 inline void update_parameters(
		        vector<float> &hidden_biases,
		        vector<float> &visible_biases,
		        vector<float> &diff_visible_biases,
		        vector<float> &diff_hidden_biases,
		        const int number_of_samples
		        ){
		            //si precalcola il fattore moltiplicativo
		            //dovendo fare una media bisogna dividere per il numero di esempi
		            const int mult_factor = rbm_learning_rate / number_of_samples;

		            //diff per pesi e bias visibili
		            for(uint i = 0; i < visible_biases.size(); i++)
		            {
		                visible_biases.at(i) += diff_visible_biases.at(i) * mult_factor;

		                diff_visible_biases.at(i) = diff_visible_biases.at(i) * rbm_momentum / mult_factor; //inizializzazione per il momentum
		            }

		            for(uint j = 0; j < hidden_biases.size(); j++){
		                hidden_biases.at(j) += diff_hidden_biases.at(j) * mult_factor;

		                diff_hidden_biases.at(j) = diff_hidden_biases.at(j) * rbm_momentum / mult_factor;//inizializzazione per il momentum
		            }
		    }


		 vector<float> reconstruct(){}
		 		void save_parameters(){}
		 	    void load_parameters(){}
	};

}
#endif /* NODE_ACCUMULATOR_AUTOENCODER_H_ */
