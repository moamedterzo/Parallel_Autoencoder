/*
 * node_accumulator_autoencoder.h
 *
 *  Created on: 19 nov 2020
 *      Author: giovanni
 */

#ifndef NODE_ACCUMULATOR_AUTOENCODER_H_
#define NODE_ACCUMULATOR_AUTOENCODER_H_


#include "node_autoencoder.h"


#include <sstream>
#include <fstream>
#include <cassert>

using namespace std;


namespace parallel_autoencoder{

	class node_accumulator_autoencoder : public node_autoencoder{

	private:
		vector<vector<float>> layer_biases;

		int k_number;

		//comunicatore da master a nodi accumulatori
		MP_Comm_MasterSlave master_accs_comm;
		vector<MP_Comm_MasterSlave> acc_rows_comm;
		vector<MP_Comm_MasterSlave> acc_cols_comm;


		vector<vector<MP_Comm_MasterSlave>> acc_hid_comm_for_layer;
		vector<vector<MP_Comm_MasterSlave>> acc_vis_comm_for_layer;
	public:

		node_accumulator_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
				int _total_accumulators, int _grid_row, int _grid_col,
				std::ostream& _oslog, int _mpi_rank,
				int _k_number,
				MP_Comm_MasterSlave& _master_accs_comm, vector<MP_Comm_MasterSlave>& _acc_rows_comm, vector<MP_Comm_MasterSlave>& _acc_cols_comm)
		: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
		{
			k_number = _k_number;

			layer_biases = vector<vector<float>>(number_of_final_layers - 1); //tutti i layer hanno il bias tranne quello di input

			master_accs_comm = _master_accs_comm;
			acc_rows_comm = _acc_rows_comm;
			acc_cols_comm = _acc_cols_comm;

			//si determinano quali comunicatori bisogna utilizzare ad ogni passo a seconda dell'orientamento scelto
			acc_hid_comm_for_layer = vector<vector<MP_Comm_MasterSlave>>(orientation_grid.size());
			acc_vis_comm_for_layer = vector<vector<MP_Comm_MasterSlave>>(orientation_grid.size());
			for(uint i = 0; i < orientation_grid.size(); i++)
			{
				const GridOrientation orientation_for_vis = orientation_grid[i];
				const GridOrientation orientation_for_hid = orientation_for_vis == row_first ? col_first : row_first;

				const int n_visible_units = layers_size[i];
				const int n_hidden_units = layers_size[i + 1];

				//comunicatori per nodi visibili e nascosti
				acc_vis_comm_for_layer[i] = orientation_for_vis == row_first ? acc_rows_comm : acc_cols_comm;
				acc_hid_comm_for_layer[i] = orientation_for_vis == row_first ? acc_cols_comm : acc_rows_comm;

				//calcolo delle size per comm acc verso le righe e verso le colonne
				calc_comm_sizes(orientation_for_vis, acc_vis_comm_for_layer[i], true, k_number, n_visible_units);
				calc_comm_sizes(orientation_for_hid, acc_hid_comm_for_layer[i], true, k_number, n_hidden_units);
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


		void BroadcastVector(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			//Si invia il vettore alle righe/colonne di riferimento, per ognuna si usa il broadcast
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				MPI_Ibcast(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}



		void AccumulateVector(vector<MP_Comm_MasterSlave>& comms, vector<float>& vec, MPI_Request *reqs)
		{
			int displacement = 0;
			for(uint i = 0; i < comms.size(); i++)
			{
				auto& comm = comms[i];

				/*MPI_Irecv(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						1, 0, comm.comm, reqs + i);


				int aaaa = comms.size() == 2 ? 5 : 3;
				//ricevo da altri 3
				for(int j = 2; j < aaaa;j++)
				{
						vector<float> asd = vec;
						MPI_Request *req = new MPI_Request;

						MPI_Irecv(vec.data() + displacement,
								comm.n_items_to_send, mpi_datatype_tosend,
								j, 0, comm.comm, req);
				}*/


				MPI_Ireduce(MPI_IN_PLACE, vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}


		void ReceiveFromMaster(vector<float>& vis_vec, MPI_Request *reqVis)
		{
		//  if(k_number == 0)	std::cout << "111111 Acc " << k_number << " scatter from master\n";

			MPI_Iscatterv(NULL, nullptr, nullptr, mpi_datatype_tosend,
					vis_vec.data(), vis_vec.size(), mpi_datatype_tosend,
					0, master_accs_comm.comm, reqVis);
		}

		void SendToMaster(vector<float>& hid_vec, MPI_Request *reqHid)
		{
			//if(k_number == 0)	std::cout << "Acc " << k_number << " invio H a master\n";

			MPI_Igatherv(hid_vec.data(), hid_vec.size(), mpi_datatype_tosend,
				MPI_IN_PLACE, nullptr, nullptr , mpi_datatype_tosend,
				0, master_accs_comm.comm, reqHid);
		}


		void get_my_visible_hidden_units(const int layer_number, int& n_my_visible_units, int& n_my_hidden_units)
		{
			//calcolo del numero di nodi visibili o nascosti rappresentati per l'accumulatore corrente
			const int n_visible_units = layers_size[layer_number];
			const int n_hidden_units = layers_size[layer_number + 1];

			n_my_visible_units = get_units_for_node(n_visible_units, total_accumulators, k_number);
			n_my_hidden_units = get_units_for_node(n_hidden_units, total_accumulators, k_number);
		}


		void train_rbm()
		{
			//1. Si apprendono le RBM per ciascun layer
			//se sono state già apprese delle rbm, si passa direttamente alla prima da imparare
			for(int layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
			{
				const bool first_layer = layer_number == 0;
				const int index_reverse_layer = number_of_final_layers - layer_number - 2;

				//comunicatori per nodi visibili e nascosti
				auto& comms_for_vis = acc_vis_comm_for_layer[layer_number];
				auto& comms_for_hid = acc_hid_comm_for_layer[layer_number];

				int n_my_visible_units, n_my_hidden_units;
				get_my_visible_hidden_units(layer_number,n_my_visible_units, n_my_hidden_units);


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
				float current_learning_rate;

				// A0) Sync ricevi input V 1 da nodo master
				ReceiveFromMaster(visible_units1, reqsVis1);
				//MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

				// A1) Async Invio V 1
				//BroadcastVector(comms_for_vis, visible_units1, reqsVis1);

				// B0) Async Ricevo secondo V (utilizzo il request dei visibili)
				//ReceiveFromMaster(visible_units2, reqsVis2);

				for(int epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					current_learning_rate = GetRBMLearningRate(epoch, layer_number);

					while(current_index_sample < (epoch + 1) * number_of_samples)
					{
						if(current_index_sample % 10 == 0 && k_number == 0)
							std::cout << "[Acc] current_index_sample: " + to_string(current_index_sample) + "\n";


						current_index_sample+=2; //ogni ciclo while gestisce due esempi

						//CONTRASTIVE DIVERGENCE
						//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
						//E' necessario però analizzare due esempi per volta


						// INIZIO SECONDO METODO
						//

						// A0) Wait ricezione input V 1 da nodo master
						MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

						// A1) Async Invio V 1
						BroadcastVector(comms_for_vis, visible_units1, reqsVis1);



						//B0) Async Ricevo secondo V (utilizzo il request dei visibili)
						ReceiveFromMaster(visible_units2, reqsVis2);

						// A1) Wait invio V1
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

						 // B0) Wait ricezione input V 2 da master
						MPI_Wait(reqsVis2, MPI_STATUS_IGNORE);

						//print_vector(visible_units2);
						//std::cout << "visible\n\n";

						// B1) invio V 2
						BroadcastVector(comms_for_vis, visible_units2, reqsVis2);
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);



						// A2) ricevo H 1
						AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < hidden_units1.size(); i++)
							hidden_units1.at(i) =
									sample_sigmoid_function(hidden_units1.at(i) + hidden_biases.at(i), generator);

						// A3) Invio H 1
						BroadcastVector(comms_for_hid, hidden_units1, reqsHid1);
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);



						// B2) ricevo H 2
						AccumulateVector(comms_for_hid, hidden_units2, reqsHid2);
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < hidden_units2.size(); i++)
							hidden_units2.at(i) =
									sample_sigmoid_function(hidden_units2.at(i)+ hidden_biases.at(i), generator);

						// B3) Invio H 2
						BroadcastVector(comms_for_hid, hidden_units2, reqsHid2);
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);



						// A4) Async ricezione Vrec 1
						AccumulateVector(comms_for_vis, rec_visible_units1, reqsVis1);
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

						//funzione Vrec 1
						//non si applica il campionamento
						if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
							for(uint i = 0; i < rec_visible_units1.size(); i++)
								rec_visible_units1.at(i) =
											sample_gaussian_distribution(rec_visible_units1.at(i) + visible_biases.at(i), generator);
						else
							for(uint i = 0; i < rec_visible_units1.size(); i++)
								rec_visible_units1.at(i) =
										sigmoid(rec_visible_units1.at(i)+ visible_biases.at(i));

						// A5) invio  Vrec 1
						BroadcastVector(comms_for_vis, rec_visible_units1, reqsVis1);
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);



						// B4) ricezione V rec 2
						AccumulateVector(comms_for_vis, rec_visible_units2, reqsVis2);
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

						//funzione Vrec 2
						//non si applica il campionamento
						if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
							for(uint i = 0; i < rec_visible_units2.size(); i++)
								rec_visible_units2.at(i) =
										sample_gaussian_distribution(rec_visible_units2.at(i)+ visible_biases.at(i), generator);
						else
							for(uint i = 0; i < rec_visible_units2.size(); i++)
								rec_visible_units2.at(i) = sigmoid(rec_visible_units2.at(i)+ visible_biases.at(i));

						// B5) invio V rec 2
						BroadcastVector(comms_for_vis, rec_visible_units2, reqsVis2);
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);




						// A6) ricezione H rec 1
						AccumulateVector(comms_for_hid, rec_hidden_units1, reqsHid1);
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < rec_hidden_units1.size(); i++)
							   rec_hidden_units1.at(i) =
									   sigmoid(rec_hidden_units1.at(i) + hidden_biases.at(i));

						// A7) invio H rec 1
						BroadcastVector(comms_for_hid, rec_hidden_units1, reqsHid1);
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);



						// B6) ricezione H rec 2
						AccumulateVector(comms_for_hid, rec_hidden_units2, reqsHid2);
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < rec_hidden_units2.size(); i++)
							   rec_hidden_units2.at(i) =
									   sigmoid(rec_hidden_units2.at(i) + hidden_biases.at(i));

						// B7) Async invio H rec 2
						BroadcastVector(comms_for_hid, rec_hidden_units2, reqsHid2);

						//gradiente 1
						//si calcolano i differenziali
						//dei pesi e bias visibili
						for(uint i = 0; i < visible_units1.size(); i++)
							diff_visible_biases.at(i) += visible_units1.at(i) - rec_visible_units1.at(i);

						//dei bias nascosti
						for(uint j = 0; j < hidden_units1.size(); j++)
							diff_hidden_biases.at(j) += hidden_units1.at(j) - rec_hidden_units1.at(j);

						// B7) Wait invio H rec 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);



						// A0) Async ricezione nuovo input V1 da master (sempre se ci sono ancora esempi)
						const bool other_samples = current_index_sample < rbm_n_training_epocs * number_of_samples;
						if(other_samples)
						{
							ReceiveFromMaster(visible_units1, reqsVis1);
						}


						//gradiente 2
						//si calcolano i differenziali
						//dei pesi e bias visibili
						for(uint i = 0; i < visible_units2.size(); i++)
							diff_visible_biases.at(i) += visible_units2.at(i) - rec_visible_units2.at(i);

						//dei bias nascosti
						for(uint j = 0; j < hidden_units2.size(); j++)
							diff_hidden_biases.at(j) += hidden_units2.at(j) - rec_hidden_units2.at(j);


						//


/*

						// A2) Async ricevo H 1
						AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);

						// A1) Wait Invio V 1
						std::cout << "[Acc" << k_number << ", wait invio V1];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);


						// B0) Wait ricezione input V 2 da master
						std::cout << "[Acc" << k_number << ", wait ricezione master V2];\n";
						MPI_Wait(reqsVis2, MPI_STATUS_IGNORE);


						// B1) Async invio V 2
						BroadcastVector(comms_for_vis, visible_units2, reqsVis2);

						// A2) Wait ricezione H 1
						std::cout << "[Acc" << k_number << ", wait ricezione H1];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						// B2) Async ricevo H 2
						AccumulateVector(comms_for_hid, hidden_units2, reqsHid2);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < hidden_units1.size(); i++)
							hidden_units1.at(i) = sample_sigmoid_function(hidden_units1.at(i), generator);


						// B1) Wait Invio V 2
						std::cout << "[Acc" << k_number << ", wait invio V2];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);


						// A3) Async Invio H 1
						BroadcastVector(comms_for_hid, hidden_units1, reqsHid1);

						// B2) Wait ricezione H 2
						std::cout << "[Acc" << k_number << ", wait ricezione H2];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);
						std::cout << "[Acc" << k_number << ", OK wait ricezione H2];\n";


						// A4) Async ricezione Vrec 1
						AccumulateVector(comms_for_vis, rec_visible_units1, reqsVis1);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < hidden_units2.size(); i++)
							hidden_units2.at(i) = sample_sigmoid_function(hidden_units2.at(i), generator);

						// A3) Wait invio H 1
						std::cout << "[Acc" << k_number << ", wait invio H1];\n";
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);
						std::cout << "[Acc" << k_number << ", OK wait invio H1];\n";


						// B3) Async Invio H 2
						BroadcastVector(comms_for_hid, hidden_units2, reqsHid2);

						// A4) Wait ricezione Vrec 1
						std::cout << "[Acc" << k_number << ", wait ricezione Vrec1];\n";
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

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
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						// A5) Async invio  Vrec 1
						BroadcastVector(comms_for_vis, rec_visible_units1, reqsVis1);

						// B4) Wait ricezione V rec 2
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

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
						MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

						// B5) Async invio V rec 2
						BroadcastVector(comms_for_vis, rec_visible_units2, reqsVis2);

						// A6) Wait ricezione H rec 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

						// B6) Async ricezione H rec 2
						AccumulateVector(comms_for_hid, rec_hidden_units2, reqsHid2);

						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < rec_hidden_units1.size(); i++)
							   rec_hidden_units1.at(i) = sigmoid(rec_hidden_units1.at(i));

						// B5) Wait invio V rec 2
						MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

						// A7) Async invio H rec 1
						BroadcastVector(comms_for_hid, rec_hidden_units1, reqsHid1);

						// B6) Wait ricezione H rec 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						//Accumulazione e sigmoide per H2
						for(uint i = 0; i < rec_hidden_units2.size(); i++)
							rec_hidden_units2.at(i) = sigmoid(rec_hidden_units2.at(i));

						// A7) Wait invio H rec 1
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

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

						std::cout << "Acc Gradienti calcolati\n";

						// B7) Wait invio H rec 2
						MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

						if(other_samples)
						{
							// A0) Wait ricezione nuovo input V1
							MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

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
*/


						//applicazione gradienti
						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0){
							/*if(k_number == 0)
							{
								print_vector(diff_hidden_biases);
													std::cout << "diff biases hidden\n\n";
													print_vector(hidden_biases);
																		std::cout << "biases hidden\n\n";
							}*/


							update_parameters(current_learning_rate,
									hidden_biases, visible_biases,
									diff_visible_biases, diff_hidden_biases, rbm_size_minibatch);

						}

					} //fine esempio
				} //fine epoca


				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0){

					//modifica pesi
					update_parameters(current_learning_rate,
							hidden_biases, visible_biases,
							diff_visible_biases, diff_hidden_biases, n_last_samples);
				}


				//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)

				for(current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					//cercare di uniformare il codice

					//0) ricevi input V da nodo master
					ReceiveFromMaster(visible_units1, reqsVis1);
					MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

					if(current_index_sample > 0)
					{
						// A2) Wait ricevo H passo precedente
						MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);
					}


					// A1) Async Invio V 1 (non importa la wait)
					BroadcastVector(comms_for_vis, visible_units1, reqsVis2);

					// A2) ricevo H 1
					AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);

					if(current_index_sample > 0)
					{
						//Accumulazione e sigmoide per H1
						for(uint i = 0; i < hidden_units1.size(); i++)
							hidden_units1.at(i) = sigmoid(hidden_units1.at(i) + hidden_biases.at(i));

						// A3) Invio H 1 a master
						SendToMaster(hidden_units1, reqsHid2);
						MPI_Wait(reqsHid2, MPI_STATUSES_IGNORE);
					}

					/*
					// A0) Wait ricezione input V 1 da nodo master
					MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

					// A1) Async Invio V 1
					BroadcastVector(comms_for_vis, visible_units1, reqsVis1);


					//B0) Async Ricevo secondo V (utilizzo il request dei visibili)
					ReceiveFromMaster(visible_units2, reqsVis2);

					// A1) Wait invio V1
					MPI_Waitall(comms_for_vis.size(), reqsVis1, MPI_STATUSES_IGNORE);

					 // B0) Wait ricezione input V 2 da master
					MPI_Wait(reqsVis2, MPI_STATUS_IGNORE);

					// B1) invio V 2
					BroadcastVector(comms_for_vis, visible_units2, reqsVis2);
					MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);



					// A2) ricevo H 1
					AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);
					MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

					//Accumulazione e sigmoide per H1
					for(uint i = 0; i < hidden_units1.size(); i++)
						hidden_units1.at(i) = sample_sigmoid_function(hidden_units1.at(i), generator);

					// A3) Invio H 1 a master
					SendToMaster(hidden_units1, reqsHid1);
					MPI_Wait(reqsHid1, MPI_STATUSES_IGNORE);



					// B2) ricevo H 2
					AccumulateVector(comms_for_hid, hidden_units2, reqsHid2);
					MPI_Waitall(comms_for_hid.size(), reqsHid2, MPI_STATUSES_IGNORE);

					//Accumulazione e sigmoide per H2
					for(uint i = 0; i < hidden_units2.size(); i++)
						hidden_units2.at(i) = sample_sigmoid_function(hidden_units2.at(i), generator);

					// B3) Invio H 2 a master
					BroadcastVector(comms_for_hid, hidden_units2, reqsHid2);
					MPI_Wait(reqsHid1, MPI_STATUSES_IGNORE);
*/


					/*// 0) Attendo ricezione vettore Visibile
					std::cout << "Waiting ric visible "<< k_number << "\n";
					MPI_Wait(reqsVis1, MPI_STATUS_IGNORE);

					//1) Async Invio V (si utilizza un altro buffer)
					visible_units2 = visible_units1;
					BroadcastVector(comms_for_vis, visible_units2, reqsVis2);

					//2) Async ricevo H
					AccumulateVector(comms_for_hid, hidden_units1, reqsHid1);

					//1) Wait Invio V
					std::cout << "Waiting invio V "<< k_number << "\n";
					MPI_Waitall(comms_for_vis.size(), reqsVis2, MPI_STATUSES_IGNORE);

					//2) Wait Ricezione H
					std::cout << "Waiting ricezione H "<< k_number << "\n";
					MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);



					//funzione sigmoide su H
					for(uint i = 0; i < hidden_units1.size(); i++)
						hidden_units1.at(i) = sigmoid(hidden_units1.at(i));

					// 4) Wait invio H a master
					//if(current_index_sample > 0)
					//	MPI_Wait(reqsHid2, MPI_STATUS_IGNORE);

					// 4) Invio async H a master (un altro buffer)
					hidden_units2 = hidden_units1;

					MPI_Igatherv(hidden_units2.data(), hidden_units2.size(), mpi_datatype_tosend,
							MPI_IN_PLACE, nullptr, nullptr , mpi_datatype_tosend,
							0, master_accs_comm.comm, reqsHid2);

					MPI_Wait(reqsHid2, MPI_STATUS_IGNORE);

					//0) Async ricevi input V da nodo master
									const bool other_samples = current_index_sample != number_of_samples - 1;
									if(other_samples)
										ReceiveFromMaster(visible_units1, reqsVis1);*/
				}

				// A2) Wait ricevo H
				MPI_Waitall(comms_for_hid.size(), reqsHid1, MPI_STATUSES_IGNORE);

				//Accumulazione e sigmoide per H1
				for(uint i = 0; i < hidden_units1.size(); i++)
					hidden_units1.at(i) = sigmoid(hidden_units1.at(i) + hidden_biases.at(i));

				// A3) Invio H 1 a master
				SendToMaster(hidden_units1, reqsHid2);
				MPI_Wait(reqsHid2, MPI_STATUSES_IGNORE);

				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();

			} //fine layer

		}

		//dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
		//la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
		 inline void update_parameters(
				const float current_learning_rate,
				vector<float> &hidden_biases,
				vector<float> &visible_biases,
				vector<float> &diff_visible_biases,
				vector<float> &diff_hidden_biases,
				const int number_of_samples
				){
					//si precalcola il fattore moltiplicativo
					//dovendo fare una media bisogna dividere per il numero di esempi
					const float mult_factor = current_learning_rate / number_of_samples;

					//diff per pesi e bias visibili
					for(uint i = 0; i < visible_biases.size(); i++)
					{
						visible_biases.at(i) += diff_visible_biases.at(i) * mult_factor;

						diff_visible_biases.at(i) = diff_visible_biases.at(i) * rbm_momentum; //inizializzazione per il momentum
					}

					for(uint j = 0; j < hidden_biases.size(); j++){
						hidden_biases.at(j) += diff_hidden_biases.at(j) * mult_factor;

						diff_hidden_biases.at(j) = diff_hidden_biases.at(j) * rbm_momentum;//inizializzazione per il momentum
					}
			}



		vector<vector<float>> get_activation_layers()
		{
			vector<vector<float>> activation_layers(number_of_final_layers);
			for(uint l = 0; l < activation_layers.size(); l++)
			{
				//calcolo del numero di nodi visibili o nascosti rappresentati per l'accumulatore corrente
				int n_my_visible_units, _;
				get_my_visible_hidden_units(l, n_my_visible_units, _);

				activation_layers.at(l) = vector<float>(n_my_visible_units);
			}

			return activation_layers;
		}


		 inline void forward_pass(vector<vector<float>>& activation_layers)
		 {
			const int central_layer = number_of_final_layers / 2 - 1;

			 //1. forward pass
			for(int l = 0; l < number_of_final_layers - 1; l++){

				//comunicatori
				auto& comms_for_vis = acc_vis_comm_for_layer[l];
				auto& comms_for_hid = acc_hid_comm_for_layer[l];

				//bias, input e output vectors
				auto& biases = layer_biases.at(l);
				auto& input = activation_layers.at(l);
				auto& output = activation_layers.at(l + 1);

				//invio a celle
				MPI_Request reqs_vis[comms_for_vis.size()];
				BroadcastVector(comms_for_vis, input, reqs_vis);

				//ricezione da celle
				MPI_Request hid_requests[comms_for_hid.size()];
				AccumulateVector(comms_for_hid, output, hid_requests);
				MPI_Waitall(comms_for_hid.size(), hid_requests, MPI_STATUSES_IGNORE);

				//si applica la funzione sigmoide
				//se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
				//per ottenere un valore binario
				if(l == central_layer)
					for(uint i = 0; i < output.size(); i++)
						output.at(i) = round(sigmoid(output.at(i) + biases.at(i)));
				else
					for(uint i = 0; i < output.size(); i++)
						output.at(i) = sigmoid(output.at(i) + biases.at(i));

			} //fine forward
		 }

		void fine_tuning()
		{
			//Rollup per i bias già effettuato in fase di apprendimento delle rbm


			//INIZIO FINE TUNING

			//si riserva lo spazio necessario per l'attivazione di ogni layer
			//e per i vettori che conterranno i valori delta per la back propagation
			auto activation_layers = get_activation_layers();


			//si esclude il primo layer dato che non possiede pesi da aggiornare
			vector<vector<float>> delta_layers(number_of_final_layers - 1);
			for(uint l = 0; l < delta_layers.size(); l++)
				delta_layers.at(l) = vector<float>(layers_size.at(l + 1));


			//unità visibili
			auto visible_units1 = vector<float>(layers_size[0]);

			//0) ricevi input V da nodo master
			MPI_Request reqMaster;
			ReceiveFromMaster(visible_units1, &reqMaster);


			//per ogni epoca...
			for(int epoch = 0; epoch < fine_tuning_n_training_epocs; epoch++)
			{
				for(int current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					//0) Wait ricevi input V da nodo master
					MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);

					//copia nel buffer
					activation_layers[0] = visible_units1;


					bool other_samples = current_index_sample != number_of_samples - 1;
					if(other_samples)
					{
						//0) ricevi input V da nodo master
						ReceiveFromMaster(visible_units1, &reqMaster);
					}

					//1. forward pass
					forward_pass(activation_layers);

					//2. backward pass
					//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
					for(int l = number_of_final_layers - 1; l >= 1; l--){

						//comunicatori
						auto& comms_for_vis = acc_vis_comm_for_layer[l];
						auto& comms_for_hid = acc_hid_comm_for_layer[l - 1];

						//si aggiornano i pesi tra l'output e l'input layer
						auto& biases_to_update = layer_biases.at(l - 1);

						//layer di attivazione
						auto& output_layer = activation_layers.at(l);

						//vettore contenente i delta
						auto current_deltas = vector<float>(output_layer.size(), 0.0);

						//check misure
						assert(output_layer.size() == biases_to_update.size());

						//si calcoleranno i delta per il layer corrente
						if(l == number_of_final_layers - 1)
						{
							//LAYER DI OUTPUT
							auto& first_activation_layer = activation_layers.at(0);

							//calcolo dei delta per il layer di output
							// delta = y_i * (1 - y_i) * reconstruction_error
							for(uint j = 0; j < output_layer.size(); j++)
							{
							   current_deltas[j] = output_layer.at(j)  * (1 - output_layer.at(j)) //derivative
									   * (first_activation_layer.at(j) - output_layer.at(j)); //rec error
							}
						}
						else
						{
							//layer nascosto
							//attendo ricezione degli input dalle celle
							//i delta vengono mandati già pesati
							MPI_Request reqs_visible[comms_for_vis.size()];
							AccumulateVector(comms_for_vis, current_deltas, reqs_visible);
							MPI_Waitall(comms_for_vis.size(), reqs_visible, MPI_STATUSES_IGNORE);

							//è necessario moltiplicare i delta per la derivata
							for(uint j = 0; j < output_layer.size(); j++)
								current_deltas[j] = current_deltas[j] * output_layer.at(j)  * (1 - output_layer.at(j)); //derivative
						}

						//Si inviano i delta alle celle (non importa aspettare)
						MPI_Request req_hid[comms_for_hid.size()];
						BroadcastVector(comms_for_hid, current_deltas, req_hid);


						//seguendo la delta rule, si applica il gradiente anche i bias
						for(uint j = 0; j < biases_to_update.size(); j++){
							biases_to_update.at(j) +=
										fine_tuning_learning_rate
										* current_deltas.at(j);
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

			auto activation_layers = get_activation_layers();
			MPI_Request reqMaster;

			//0) Wait ricevi input V da nodo master
			ReceiveFromMaster(activation_layers[0], &reqMaster);
			MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);

			//1. forward pass
			forward_pass(activation_layers);


			//invio vettore a master
			auto& last_layer = activation_layers[number_of_final_layers - 1];
			SendToMaster(last_layer, &reqMaster);
			MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);


			return last_layer;//dummy
		}




	   string get_path_file(){

			return folder_parameters_path + "paral_k_"+ std::to_string(k_number) + ".txt";
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
				int index_reverse_layer = number_of_final_layers - layer_number - 2;

				auto& hidden_biases = layer_biases[layer_number];
				auto& visible_biases =  layer_biases[index_reverse_layer];


				myFile << "_hidden_" << hidden_biases.size() << "__,";
				for(auto& v : hidden_biases)
					myFile << fixed << setprecision(F_PREC) << v << ",";

				myFile << endl;

				myFile << "_visible_" << visible_biases.size() << "__,";
				for(auto& v : visible_biases)
					myFile << fixed << setprecision(F_PREC) << v << ",";
				myFile << endl;
			}

			if(fine_tuning_finished)
			{
				while(layer_number < number_of_final_layers - 1)
				{
					auto& hidden_biases = layer_biases[layer_number];

					myFile << "_rec_" << hidden_biases.size() << "__,";
					for(auto& v : hidden_biases)
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

			vector<float> *current_hidden_biases;
			vector<float> *current_visible_biases;

			int current_row_file = 0;

			//Si leggono le linee che contengono i pesi delle rbm apprese
			bool other_lines = false;
			while(std::getline(myFile, line))
			{
				int n_my_visible_units, n_my_hidden_units;

				if(current_row_file % 2 == 0)
				{
					//se abbiamo letto tutti i pesi delle rbm si esce da questo ciclo
					if(trained_rbms == number_of_rbm_to_learn){
						//se ci sono altre linee, vuol dire che si possiedono i parametri dei layer di ricostruzione
						other_lines = true;
						break;
					}

					//si ottengono grandezze
					get_my_visible_hidden_units(trained_rbms, n_my_visible_units, n_my_hidden_units);
					layer_biases.at(trained_rbms) = vector<float>(n_my_hidden_units);

					int index_reverse_layer = number_of_final_layers - trained_rbms - 2;

					layer_biases.at(index_reverse_layer) = vector<float>(n_my_visible_units); //per simmetria delle rete è corretto

					current_hidden_biases = &layer_biases.at(trained_rbms);
					current_visible_biases = &layer_biases.at(index_reverse_layer);

					//questa variabile memorizza il numero di rbm apprese
					trained_rbms++;
				}

				// Create a stringstream of the current line
				std::stringstream ss(line);

				//in base alla riga si aggiornano i relativi parametri
				if (current_row_file % 2 == 0){

					//riga dei bias nascosti
					ss.ignore(100, ',');
					for(uint j = 0; j < n_my_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->at(j);
					}
				}
				else if (current_row_file % 2 == 1){

					//riga dei bias visibili
					ss.ignore(100, ',');
					for(uint i = 0; i < n_my_visible_units; i++) {
						if(ss.peek() == ',') ss.ignore();
						ss >> current_visible_biases->at(i);
					}
				}

				//si tiene conto della riga processata
				current_row_file++;
			}


			//se ci sono altre linee da analizzare vuol dire che si aggiornano i pesi dei layer di ricostruzione
			if(other_lines)
			{
				current_row_file = 0;

				//indice del layer contenente pesi o bias
				int current_layer = (number_of_final_layers - 1) / 2;
				do
				{
					//c'è un solo layer per i bias da aggiornare
					int n_my_visible_units, n_my_hidden_units;
					get_my_visible_hidden_units(current_layer, n_my_visible_units, n_my_hidden_units);

					current_hidden_biases = &layer_biases.at(current_layer); //già inizializzato

					//check sulle misure
					assert(n_my_hidden_units == current_hidden_biases->size());


					current_layer++;

					// Create a stringstream of the current line
					std::stringstream ss(line);


					//riga dei bias
					ss.ignore(100, ',');
					for(uint j = 0; j < n_my_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->at(j);
					}


					//si tiene conto della riga processata
					current_row_file++;
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
#endif /* NODE_ACCUMULATOR_AUTOENCODER_H_ */
