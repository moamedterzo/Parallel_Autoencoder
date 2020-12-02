/*
 * node_accumulator_autoencoder.cpp
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#include "node_accumulator_autoencoder.h"

#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;


namespace parallel_autoencoder
{

	inline void set_gradient(my_vector<float>& diff_biases,
				my_vector<float>& units,
				my_vector<float>& rec_units)
		{
			for(uint i = 0; i != diff_biases.size(); i++)
				diff_biases[i] += units[i] - rec_units[i];
		}





		void node_accumulator_autoencoder::calc_all_comm_sizes()
		{
			//si determinano quali comunicatori bisogna utilizzare ad ogni passo a seconda dell'orientamento scelto
			acc_hid_comm_for_layer = my_vector<my_vector<MP_Comm_MasterSlave>>(orientation_grid.size());
			acc_vis_comm_for_layer = my_vector<my_vector<MP_Comm_MasterSlave>>(orientation_grid.size());

			for(uint i = 0; i < orientation_grid.size(); i++)
			{
				const GridOrientation orientation_for_vis = orientation_grid[i];
				const GridOrientation orientation_for_hid = orientation_for_vis == GridOrientation::row_first ?
						GridOrientation::col_first : GridOrientation::row_first;

				const uint n_visible_units = layers_size[i];
				const uint n_hidden_units = layers_size[i + 1];

				//comunicatori per nodi visibili e nascosti
				acc_vis_comm_for_layer[i] = orientation_for_vis == GridOrientation::row_first ? acc_rows_comm : acc_cols_comm;
				acc_hid_comm_for_layer[i] = orientation_for_vis == GridOrientation::row_first ? acc_cols_comm : acc_rows_comm;

				//calcolo delle size per comm acc verso le righe e verso le colonne
				calc_comm_sizes(orientation_for_vis, acc_vis_comm_for_layer[i], true, k_number, n_visible_units);
				calc_comm_sizes(orientation_for_hid, acc_hid_comm_for_layer[i], true, k_number, n_hidden_units);
			}
		}


		void node_accumulator_autoencoder::ReceiveFromMaster(my_vector<float>& vis_vec, MPI_Request *reqVis)
		{
			MPI_Iscatterv(NULL, nullptr, nullptr, mpi_datatype_tosend,
					vis_vec.data(), vis_vec.size(), mpi_datatype_tosend,
					0, master_accs_comm, reqVis);
		}

		void node_accumulator_autoencoder::SendToMaster(my_vector<float>& hid_vec, MPI_Request *reqHid)
		{
			MPI_Igatherv(hid_vec.data(), hid_vec.size(), mpi_datatype_tosend,
				MPI_IN_PLACE, nullptr, nullptr , mpi_datatype_tosend,
				0, master_accs_comm, reqHid);
		}


		void node_accumulator_autoencoder::get_my_visible_hidden_units(const uint layer_number, uint& n_my_visible_units, uint& n_my_hidden_units)
		{
			//calcolo del numero di nodi visibili o nascosti rappresentati per l'accumulatore corrente
			const int n_visible_units = layers_size[layer_number];
			const int n_hidden_units = layers_size[layer_number + 1];

			n_my_visible_units = get_units_for_node(n_visible_units, total_accumulators, k_number);
			n_my_hidden_units = get_units_for_node(n_hidden_units, total_accumulators, k_number);
		}


		node_accumulator_autoencoder::node_accumulator_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
						uint _total_accumulators, uint _grid_row, uint _grid_col,
						uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode,
						std::ostream& _oslog, uint _mpi_rank,
						uint _k_number,
						MPI_Comm& _master_accs_comm,
						my_vector<MP_Comm_MasterSlave>& _acc_rows_comm, my_vector<MP_Comm_MasterSlave>& _acc_cols_comm)

				: node_autoencoder(_layers_size, _generator, _total_accumulators,  _grid_row, _grid_col, rbm_n_epochs, finetuning_n_epochs, batch_mode, _oslog, _mpi_rank)

				{
					k_number = _k_number;

					//tutti i layer hanno il bias tranne quello di input
					layer_biases = my_vector<my_vector<float>>(number_of_final_layers - 1);

					master_accs_comm = _master_accs_comm;
					acc_rows_comm = _acc_rows_comm;
					acc_cols_comm = _acc_cols_comm;

					calc_all_comm_sizes();
				}



		void node_accumulator_autoencoder::train_rbm()
		{
			//1. Si apprendono le RBM per ciascun layer
			//se sono state già apprese delle rbm, si passa direttamente alla prima da imparare
			for(uint layer_number = trained_rbms; layer_number != number_of_rbm_to_learn; layer_number++)
			{
				const bool first_layer = layer_number == 0;
				const uint index_reverse_layer = number_of_final_layers - layer_number - 2;

				uint n_my_visible_units, n_my_hidden_units;
				get_my_visible_hidden_units(layer_number, n_my_visible_units, n_my_hidden_units);

				//RBM
				//inizializzazione bias
				auto& hidden_biases = layer_biases[layer_number];
				auto& visible_biases =  layer_biases[index_reverse_layer];

				visible_biases = my_vector<float>(n_my_visible_units, rbm_initial_biases_value);
				hidden_biases = my_vector<float>(n_my_hidden_units, rbm_initial_biases_value);

				//layers visibili e nascosti, ricostruiti e non
				my_vector<float> visible_units1(n_my_visible_units, 0.0);
				my_vector<float> hidden_units1(n_my_hidden_units, 0.0);
				my_vector<float> rec_visible_units1(n_my_visible_units, 0.0);
				my_vector<float> rec_hidden_units1(n_my_hidden_units, 0.0);

				my_vector<float> visible_units2(n_my_visible_units, 0.0);
				my_vector<float> hidden_units2(n_my_hidden_units, 0.0);
				my_vector<float> rec_visible_units2(n_my_visible_units, 0.0);
				my_vector<float> rec_hidden_units2(n_my_hidden_units, 0.0);

				//gradienti calcolati per pesi e bias
				my_vector<float> diff_visible_biases(n_my_visible_units, 0.0);
				my_vector<float> diff_hidden_biases(n_my_hidden_units, 0.0);

				//comunicatori per nodi visibili e nascosti
				auto& comms_for_vis = acc_vis_comm_for_layer[layer_number];
				auto& comms_for_hid = acc_hid_comm_for_layer[layer_number];

				//Puntatori delle request asincrone
				MPI_Request reqMaster;
				MPI_Request reqsVis1[comms_for_vis.size()];
				MPI_Request reqsHid1[comms_for_hid.size()];

				MPReqManagerAccumulator reqVisible1{ reqsVis1, &comms_for_vis };
				MPReqManagerAccumulator reqHidden1{ reqsHid1, &comms_for_hid };



				//si avvia il processo di apprendimento per diverse epoche
				ulong current_index_sample = 0;
				float current_learning_rate;

				// A0) Sync ricevi input V 1 da nodo master
				ReceiveFromMaster(visible_units1, &reqMaster);

				for(uint epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					if(k_number == 0) std::cout << "Training epoch: " << epoch << "\n";

					current_learning_rate = GetRBMLearningRate(epoch, layer_number);

					while(current_index_sample < (epoch + 1) * number_of_samples)
					{
						current_index_sample+=2; //ogni ciclo while gestisce due esempi

						//CONTRASTIVE DIVERGENCE
						//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
						//E' necessario però analizzare due esempi per volta


						// A0) Wait ricezione input V 1 da nodo master
						MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);

						// A1) Async Invio V 1
						reqVisible1.BroadcastVector(visible_units1);



						//B0) Async Ricevo secondo V (utilizzo il request dei visibili)
						ReceiveFromMaster(visible_units2, &reqMaster);

						// A1) Wait invio V1
						reqVisible1.wait();

						 // B0) Wait ricezione input V 2 da master
						MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);

						// B1) invio V 2
						reqVisible1.BroadcastVectorSync(visible_units2);



						// A2) ricevo H 1
						reqHidden1.AccumulateVectorSync(hidden_units1);

						//Accumulazione e sigmoide per H1
						SampleHiddenUnits(hidden_units1, hidden_biases, generator);

						// A3) Invio H 1
						reqHidden1.BroadcastVectorSync(hidden_units1);


						// B2) ricevo H 2
						reqHidden1.BroadcastVectorSync(hidden_units2);

						//Accumulazione e sigmoide per H2
						SampleHiddenUnits(hidden_units2, hidden_biases, generator);

						// B3) Invio H 2
						reqHidden1.BroadcastVectorSync(hidden_units2);


						// A4) Async ricezione Vrec 1
						reqVisible1.AccumulateVector(rec_visible_units1);

						//funzione Vrec 1
						ReconstructVisibileUnits(rec_visible_units1, visible_biases, first_layer, generator);

						// A5) invio  Vrec 1
						reqVisible1.BroadcastVectorSync(rec_visible_units1);



						// B4) ricezione V rec 2
						reqVisible1.AccumulateVectorSync(rec_visible_units2);

						//funzione Vrec 2
						//non si applica il campionamento
						ReconstructVisibileUnits(rec_visible_units2, visible_biases, first_layer, generator);

						// B5) invio V rec 2
						reqVisible1.BroadcastVectorSync(rec_visible_units2);




						// A6) ricezione H rec 1
						reqHidden1.AccumulateVectorSync(rec_hidden_units1);

						//Accumulazione e sigmoide per H1
						ReconstructHiddenUnits(rec_hidden_units1, hidden_biases, first_layer, generator);

						// A7) invio H rec 1
						reqHidden1.BroadcastVectorSync(rec_hidden_units1);


						// B6) ricezione H rec 2
						reqHidden1.AccumulateVectorSync(rec_hidden_units2);

						//Accumulazione e sigmoide per H2
						ReconstructHiddenUnits(rec_hidden_units2, hidden_biases, first_layer, generator);

						// B7) Async invio H rec 2
						reqHidden1.BroadcastVector(rec_hidden_units2);

						//gradiente 1
						//si calcolano i differenziali
						//dei pesi e bias visibili
						set_gradient(diff_visible_biases, visible_units1, rec_visible_units1);

						//dei bias nascosti
						set_gradient(diff_hidden_biases, hidden_units1, rec_hidden_units1);

						// B7) Wait invio H rec 2
						reqHidden1.wait();


						// A0) Async ricezione nuovo input V1 da master (sempre se ci sono ancora esempi)
						const bool other_samples = current_index_sample < rbm_n_training_epocs * number_of_samples;
						if(other_samples)
							ReceiveFromMaster(visible_units1, &reqMaster);


						//gradiente 2
						//si calcolano i differenziali
						//dei pesi e bias visibili
						set_gradient(diff_visible_biases, visible_units2, rec_visible_units2);

						//dei bias nascosti
						set_gradient(diff_hidden_biases, hidden_units2, rec_hidden_units2);


						//applicazione gradienti
						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0)
							update_parameters_biases(rbm_momentum, current_learning_rate, hidden_biases, visible_biases,
									diff_visible_biases, diff_hidden_biases, rbm_size_minibatch);

					} //fine esempio
				} //fine epoca


				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0)
					update_parameters_biases(rbm_momentum, current_learning_rate, hidden_biases, visible_biases,
							diff_visible_biases, diff_hidden_biases, n_last_samples);


				//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)
				save_new_samples(reqVisible1, reqHidden1, &reqMaster,
						hidden_biases, visible_units1, hidden_units1);

				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();

			} //fine layer

		}




		 void node_accumulator_autoencoder::save_new_samples(
				 MPReqManagerAccumulator& reqVis, MPReqManagerAccumulator& reqHid,
				 MPI_Request *reqMaster, my_vector<float>& hidden_biases,
				 my_vector<float>& visible_units1, my_vector<float>& hidden_units1)
		 {
			for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
			{
				//cercare di uniformare il codice

				//0) ricevi input V da nodo master
				ReceiveFromMaster(visible_units1, reqMaster);
				MPI_Wait(reqMaster, MPI_STATUS_IGNORE);

				// A2) Wait ricevo H passo precedente
				if(current_index_sample != 0)
					reqHid.wait();


				// A1) Async Invio V 1 (non importa la wait)
				reqVis.BroadcastVector(visible_units1);

				// A2) ricevo H 1
				reqHid.AccumulateVector(hidden_units1);

				if(current_index_sample > 0)
				{
					//Accumulazione e sigmoide per H1
					for(uint i = 0; i != hidden_units1.size(); i++)
						hidden_units1[i] = sigmoid(hidden_units1[i] + hidden_biases[i]);

					// A3) Invio H 1 a master
					SendToMaster(hidden_units1, reqMaster);
					MPI_Wait(reqMaster, MPI_STATUSES_IGNORE);
				}
			}

			// A2) Wait ricevo H
			reqHid.wait();

			//Accumulazione e sigmoide per H1
			for(uint i = 0; i != hidden_units1.size(); i++)
				hidden_units1[i] = sigmoid(hidden_units1[i] + hidden_biases[i]);

			// A3) Invio H 1 a master
			SendToMaster(hidden_units1, reqMaster);
			MPI_Wait(reqMaster, MPI_STATUSES_IGNORE);
		 }



		my_vector<my_vector<float>> node_accumulator_autoencoder::get_activation_layers()
		{
			my_vector<my_vector<float>> activation_layers(number_of_final_layers);
			for(uint l = 0; l != activation_layers.size(); l++)
			{
				//calcolo del numero di nodi visibili o nascosti rappresentati per l'accumulatore corrente
				const int n_visible_units = layers_size[l];
				const uint n_my_visible_units = get_units_for_node(n_visible_units, total_accumulators, k_number);

				activation_layers[l] = my_vector<float>(n_my_visible_units, 0.0);
			}

			return activation_layers;
		}


		void node_accumulator_autoencoder::forward_pass(my_vector<my_vector<float>>& activation_layers)
		{
			const uint central_layer = number_of_final_layers / 2 - 1;

			 //1. forward pass
			for(uint l = 0; l != number_of_final_layers - 1; l++){

				//comunicatori
				auto& comms_for_vis = acc_vis_comm_for_layer[l];
				auto& comms_for_hid = acc_hid_comm_for_layer[l];

				//bias, input e output vectors
				auto& biases = layer_biases[l];
				auto& input = activation_layers[l];
				auto& output = activation_layers[l + 1];

				MPI_Request reqs_vis[comms_for_vis.size()];
				MPI_Request hid_requests[comms_for_hid.size()];

				MPReqManagerAccumulator reqVis{ reqs_vis, &comms_for_vis };
				MPReqManagerAccumulator reqHid{ hid_requests, &comms_for_hid };


				//invio a celle
				reqVis.BroadcastVector(input);

				//ricezione da celle
				reqHid.AccumulateVectorSync(output);

				//si applica la funzione sigmoide
				//se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
				//per ottenere un valore binario
				apply_sigmoid_to_layer(output, biases, l == central_layer);

			} //fine forward
		 }

		void node_accumulator_autoencoder::backward_pass(my_vector<my_vector<float>>& activation_layers)
		 {
			//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
			for(uint l = number_of_final_layers - 1; l != 0; l--){

				//comunicatori
				auto& comms_for_hid = acc_hid_comm_for_layer[l - 1];

				//si aggiornano i pesi tra l'output e l'input layer
				auto& biases_to_update = layer_biases[l - 1];

				//layer di attivazione
				auto& output_layer = activation_layers[l];

				//vettore contenente i delta
				auto current_deltas = my_vector<float>(output_layer.size(), 0.0);

				//check misure
				assert(output_layer.size() == biases_to_update.size());

				//si calcoleranno i delta per il layer corrente
				if(l == number_of_final_layers - 1)
				{
					//LAYER DI OUTPUT
					auto& first_activation_layer = activation_layers[0];

					//calcolo dei delta per il layer di output
					// delta = y_i * (1 - y_i) * reconstruction_error
					deltas_for_output_layer(output_layer, first_activation_layer, current_deltas);
				}
				else
				{
					//layer nascosto
					//attendo ricezione degli input dalle celle
					//i delta vengono mandati già pesati
					auto& comms_for_vis = acc_vis_comm_for_layer[l];

					MPI_Request reqs_visible[comms_for_vis.size()];

					MPReqManagerAccumulator reqVis{reqs_visible, &comms_for_vis};

					reqVis.AccumulateVectorSync(current_deltas);

					//è necessario moltiplicare i delta per la derivata
					for(uint j = 0; j != output_layer.size(); j++)
						current_deltas[j] = current_deltas[j] * output_layer[j]  * (1 - output_layer[j]); //derivative
				}

				//Si inviano i delta alle celle (non importa aspettare)
				MPI_Request req_hid[comms_for_hid.size()];


				MPReqManagerAccumulator reqHid{req_hid, &comms_for_hid};

				reqHid.BroadcastVector(current_deltas);

				//seguendo la delta rule, si applica il gradiente anche i bias
				update_biases_fine_tuning(biases_to_update, current_deltas, fine_tuning_learning_rate);
			}
		 }


		void node_accumulator_autoencoder::fine_tuning()
		{
			//Rollup per i bias già effettuato in fase di apprendimento delle rbm

			//INIZIO FINE TUNING

			//si riserva lo spazio necessario per l'attivazione di ogni layer
			//e per i vettori che conterranno i valori delta per la back propagation
			auto activation_layers = get_activation_layers();

			//unità visibili
			auto visible_units1 = my_vector<float>(layers_size[0]);

			//0) ricevi input V da nodo master
			MPI_Request reqMaster;
			ReceiveFromMaster(visible_units1, &reqMaster);


			//per ogni epoca e ogni suo esempio...
			for(uint epoch = 0; epoch != fine_tuning_n_training_epocs; epoch++)
			{
				if(k_number == 0 ) std::cout << "Training epoch: " << epoch << "\n";

				for(uint current_index_sample = 0;
						current_index_sample != number_of_samples;
						current_index_sample++)
				{


					//0) Wait ricevi input V da nodo master
					MPI_Wait(&reqMaster, MPI_STATUS_IGNORE);

					//copia nel buffer
					activation_layers[0] = visible_units1;


					bool other_samples = epoch != (fine_tuning_n_training_epocs - 1) || current_index_sample != number_of_samples - 1;
					if(other_samples)
					{
						//0) ricevi input V da nodo master
						ReceiveFromMaster(visible_units1, &reqMaster);
					}

					//1. forward pass
					forward_pass(activation_layers);

					//2. backward pass
					backward_pass(activation_layers);

				} //fine esempi
			}


			//allenamento concluso
			fine_tuning_finished = true;
			save_parameters();

		}



		my_vector<float> node_accumulator_autoencoder::reconstruct(){


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




		string node_accumulator_autoencoder::get_path_file(){

			return folder_parameters_path + "paral_k_"+ std::to_string(k_number) + ".txt";
		}


		void node_accumulator_autoencoder::save_parameters(){

			string path_file = get_path_file();

			// Create an input filestream
			std::ofstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);



			//salvataggio di pesi, bias
			uint layer_number;
			for(layer_number = 0; layer_number != trained_rbms; layer_number++)
			{
				uint index_reverse_layer = number_of_final_layers - layer_number - 2;


				auto& hidden_biases = layer_biases[layer_number];
				myFile << "_hidden_" << hidden_biases.size() << "__,";
				for(uint i = 0; i < hidden_biases.size(); i++)
					myFile << fixed << setprecision(F_PREC) << hidden_biases[i] << ",";

				myFile << endl;


				auto& visible_biases =  layer_biases[index_reverse_layer];
				myFile << "_visible_" << visible_biases.size() << "__,";

				for(uint i = 0; i < visible_biases.size(); i++)
					myFile << fixed << setprecision(F_PREC) << visible_biases[i] << ",";

				myFile << endl;
			}

			if(fine_tuning_finished)
			{
				while(layer_number != number_of_final_layers - 1)
				{
					auto& hidden_biases = layer_biases[layer_number];

					myFile << "_rec_" << hidden_biases.size() << "__,";

					for(uint i = 0; i < hidden_biases.size(); i++)
						myFile << fixed << setprecision(F_PREC) << hidden_biases[i] << ",";

					myFile << endl;

					layer_number++;
				}
			}

			myFile.close();

		}

		void node_accumulator_autoencoder::load_parameters(){

			string path_file = get_path_file();


			fine_tuning_finished = false;
			trained_rbms = 0;

			// Create an input filestream
			std::ifstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

			// Helper vars
			std::string line;

			my_vector<float> *current_hidden_biases;
			my_vector<float> *current_visible_biases;

			uint current_row_file = 0;

			//Si leggono le linee che contengono i pesi delle rbm apprese
			bool other_lines = false;
			while(std::getline(myFile, line))
			{
				uint n_my_visible_units, n_my_hidden_units;

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
					layer_biases[trained_rbms] = my_vector<float>(n_my_hidden_units);

					uint index_reverse_layer = number_of_final_layers - trained_rbms - 2;

					//per simmetria delle rete è corretto
					layer_biases[index_reverse_layer] = my_vector<float>(n_my_visible_units);

					current_hidden_biases = &layer_biases[trained_rbms];
					current_visible_biases = &layer_biases[index_reverse_layer];

					//questa variabile memorizza il numero di rbm apprese
					trained_rbms++;
				}

				// Create a stringstream of the current line
				std::stringstream ss(line);

				//in base alla riga si aggiornano i relativi parametri
				if (current_row_file % 2 == 0){

					//riga dei bias nascosti
					ss.ignore(100, ',');
					for(uint j = 0; j != n_my_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->operator [](j);
					}
				}
				else if (current_row_file % 2 == 1){

					//riga dei bias visibili
					ss.ignore(100, ',');
					for(uint i = 0; i != n_my_visible_units; i++) {
						if(ss.peek() == ',') ss.ignore();
						ss >> current_visible_biases->operator [](i);;
					}
				}

				//si tiene conto della riga processata
				current_row_file++;
			}


			//se ci sono altre linee da analizzare vuol dire che si aggiornano i pesi dei layer di ricostruzione
			if(other_lines)
			{
				//indice del layer contenente pesi o bias
				uint current_layer = (number_of_final_layers - 1) / 2;
				do
				{
					//c'è un solo layer per i bias da aggiornare
					uint n_my_visible_units, n_my_hidden_units;
					get_my_visible_hidden_units(current_layer, n_my_visible_units, n_my_hidden_units);

					current_hidden_biases = &layer_biases[current_layer]; //già inizializzato

					//check sulle misure
					assert(n_my_hidden_units == current_hidden_biases->size());


					current_layer++;

					// Create a stringstream of the current line
					std::stringstream ss(line);


					//riga dei bias
					ss.ignore(100, ',');
					for(uint j = 0; j != n_my_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->operator [](j);
					}
				}
				while(std::getline(myFile, line));

				//il training si considera concluso
				fine_tuning_finished = true;

			}

			// Close file
			myFile.close();

		}





}



