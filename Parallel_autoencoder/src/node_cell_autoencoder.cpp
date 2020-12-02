/*
 * node_cell_autoencoder.cpp
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#include "node_cell_autoencoder.h"

#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;


namespace parallel_autoencoder
{


	inline void set_gradient(matrix<float>& diff_weights,
			my_vector<float>& visible_units,my_vector<float>& hidden_units,
			my_vector<float>& rec_visible_units,my_vector<float>& rec_hidden_units)
	{
		for(uint i = 0; i != visible_units.size(); i++)
		{
			for(uint j = 0; j != hidden_units.size(); j++){
				diff_weights.at(i, j) +=
						  visible_units[i] * hidden_units[j]  //fattore positivo
						 - rec_visible_units[i] * rec_hidden_units[j]; //fattore negativo
			}
		}
	}



	void node_cell_autoencoder::calc_all_comm_sizes()
	{
		//si determinano quali comunicatori bisogna utilizzare ad ogni passo a seconda dell'orientamento scelto
		acc_hid_comm_for_layer = my_vector<my_vector<MP_Comm_MasterSlave>>(orientation_grid.size());
		acc_vis_comm_for_layer = my_vector<my_vector<MP_Comm_MasterSlave>>(orientation_grid.size());

		for(uint i = 0; i != orientation_grid.size(); i++)
		{
			const GridOrientation orientation_for_vis = orientation_grid[i];
			const GridOrientation orientation_for_hid = orientation_for_vis == GridOrientation::row_first ?
					GridOrientation::col_first : GridOrientation::row_first;

			//calcolo del numero di nodi visibili o nascosti rappresentati per la cella corrente
			const uint n_visible_units = layers_size[i];
			const uint n_hidden_units = layers_size[i + 1];

			//b) che posizione rappresenta la cella per il vettore visibile/nascosto
			const auto my_vis_number = orientation_for_vis == GridOrientation::row_first ? row_number : col_number;
			const auto my_hid_number = orientation_for_vis == GridOrientation::row_first ? col_number : row_number;

			//comunicatori per nodi visibili e nascosti
			acc_vis_comm_for_layer[i] = orientation_for_vis == GridOrientation::row_first ? accs_row_comm : accs_col_comm;
			acc_hid_comm_for_layer[i] = orientation_for_vis == GridOrientation::row_first ? accs_col_comm : accs_row_comm;

			//calcolo delle size per comm acc verso le righe e verso le colonne
			calc_comm_sizes(orientation_for_vis, acc_vis_comm_for_layer[i], false, my_vis_number, n_visible_units);
			calc_comm_sizes(orientation_for_hid, acc_hid_comm_for_layer[i], false, my_hid_number, n_hidden_units);
		}
	}


	void node_cell_autoencoder::get_my_visible_hidden_units(const uint layer_number, uint& n_my_visible_units, uint& n_my_hidden_units)
	{
		//orientamento grid
		const GridOrientation orientation_for_vis = orientation_grid[layer_number];

		//calcolo del numero di nodi visibili o nascosti rappresentati per la cella corrente
		const int n_visible_units = layers_size[layer_number];
		const int n_hidden_units = layers_size[layer_number + 1];

		//a) in quante parti deve essere diviso il vettore dei visibili/nascosti
		const auto total_for_vis = orientation_for_vis == GridOrientation::row_first ? grid_rows : grid_cols;
		const auto total_for_hid = orientation_for_vis == GridOrientation::row_first ? grid_cols : grid_rows;

		//b) che posizione rappresenta la cella per il vettore visibile/nascosto
		const auto my_vis_number = orientation_for_vis == GridOrientation::row_first ? row_number : col_number;
		const auto my_hid_number = orientation_for_vis == GridOrientation::row_first ? col_number : row_number;

		n_my_visible_units = get_units_for_node(n_visible_units, total_for_vis, my_vis_number);
		n_my_hidden_units = get_units_for_node(n_hidden_units, total_for_hid, my_hid_number);
	}



	node_cell_autoencoder::node_cell_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
			uint _total_accumulators, uint _grid_row, uint _grid_col,
			uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode,
			std::ostream& _oslog, int _mpi_rank,
			uint _row_number, uint _col_number,
			my_vector<MP_Comm_MasterSlave>& _accs_row_comm, my_vector<MP_Comm_MasterSlave>& _accs_col_comm)

	: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col,rbm_n_epochs, finetuning_n_epochs, batch_mode, _oslog, _mpi_rank)
	{
		row_number = _row_number;
		col_number = _col_number;

		layers_weights = my_vector<matrix<float>>(number_of_final_layers - 1);

		accs_row_comm = _accs_row_comm;
		accs_col_comm = _accs_col_comm;

		calc_all_comm_sizes();

	}


	void node_cell_autoencoder::train_rbm()
	{
		//1. Si apprendono le RBM per ciascun layer
		//se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
		for(uint layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
		{
			uint n_my_visible_units, n_my_hidden_units;
			get_my_visible_hidden_units(layer_number,n_my_visible_units, n_my_hidden_units);

			//RBM
			//la matrice dei pesi per il layer in questione,
			//possiede grandezza VxH (unità visibili per unità nascoste)
			//si riserva lo spazio necessario
			layers_weights[layer_number] = matrix<float>(n_my_visible_units, n_my_hidden_units);
			auto& weights = layers_weights[layer_number];

			//inizializzazione pesi
			initialize_weight_matrix(weights, rbm_initial_weights_mean, rbm_initial_weights_variance, generator);

			//gradienti calcolati per pesi
			matrix<float> diff_weights(n_my_visible_units, n_my_hidden_units, 0.0);


			//layers visibili e nascosti, ricostruiti e non
			my_vector<float> visible_units1(n_my_visible_units);
			my_vector<float> hidden_units1(n_my_hidden_units);
			my_vector<float> rec_visible_units1(n_my_visible_units);
			my_vector<float> rec_hidden_units1(n_my_hidden_units);

			my_vector<float> visible_units2(n_my_visible_units);
			my_vector<float> hidden_units2(n_my_hidden_units);
			my_vector<float> rec_visible_units2(n_my_visible_units);
			my_vector<float> rec_hidden_units2(n_my_hidden_units);


			//Puntatori delle request asincrone
			//comunicatori per nodi visibili e nascosti
			auto& comms_for_vis = acc_vis_comm_for_layer[layer_number];
			auto& comms_for_hid = acc_hid_comm_for_layer[layer_number];

			MPI_Request reqsVis1[comms_for_vis.size()];
			MPI_Request reqsVis1Ricezione[comms_for_vis.size()];

			MPI_Request reqsHid1[comms_for_hid.size()];
			MPI_Request reqsHid1Ricezione[comms_for_hid.size()];


			MPReqManagerCell reqVisible1{ reqsVis1, &comms_for_vis};
			MPReqManagerCell reqVisibleRicezione2{ reqsVis1Ricezione, &comms_for_vis};

			MPReqManagerCell reqHidden1{reqsHid1, &comms_for_hid };
			MPReqManagerCell reqHiddenRicezione1{reqsHid1Ricezione, &comms_for_hid };


			//si avvia il processo di apprendimento per diverse epoche
			ulong current_index_sample = 0;
			float current_learning_rate;

			// A1) Async Ricezione V 1
			reqVisible1.ReceiveVector(visible_units1);

			for(uint epoch = 0; epoch < rbm_n_training_epocs; epoch++){

				current_learning_rate = GetRBMLearningRate(epoch, layer_number);

				while(current_index_sample < (epoch + 1) * number_of_samples)
				{
					current_index_sample+=2; //ogni ciclo while gestisce due esempi

					//CONTRASTIVE DIVERGENCE
					//Si utilizza un protocollo di comunicazione che permette di effettuare computazioni mentre si inviano dati
					//E' necessario però analizzare due esempi per volta

					// A1) Wait ricezione V 1
					reqVisible1.wait();


					// B1) Async ricezione V
					reqVisible1.ReceiveVector(visible_units2);

					//Prodotto matriciale V * W
					matrix_transpose_vector_multiplication(weights, visible_units1, hidden_units1);

					// B1) Wait ricezione V
					reqVisible1.wait();




					// A2) Async invio H 1
					reqHidden1.SendVectorToReduce(hidden_units1);


					// A3) Async ricezione H
					reqHiddenRicezione1.ReceiveVector(hidden_units1);

					//Prodotto matriciale V * W
					matrix_transpose_vector_multiplication(weights, visible_units2, hidden_units2);

					// A2) Wait invio H
					reqHidden1.wait();

					// A3) Wait ricezione H
					reqHiddenRicezione1.wait();



					// B2) Async invio H
					reqHidden1.SendVectorToReduce(hidden_units2);



					// B3) Async ricezione H
					reqHiddenRicezione1.ReceiveVector(hidden_units2);

					//Prodotto matriciale H * W
					matrix_vector_multiplication(weights, hidden_units1, rec_visible_units1);

					// B2) Wait invio H
					reqHidden1.wait();


					// B3) Wait ricezione H
					reqHiddenRicezione1.wait();



					// A4) Async invio Vrec
					reqVisible1.SendVectorToReduce(rec_visible_units1);

					// A5) Async ricezione Vrec
					reqVisibleRicezione2.ReceiveVector(rec_visible_units1);

					//Prodotto matriciale H * W
					matrix_vector_multiplication(weights, hidden_units2, rec_visible_units2);

					// A4) Wait invio V rec
					reqVisible1.wait();

					// A5) Wait ricezione V rec
					reqVisibleRicezione2.wait();



					// B4) Async invio V rec
					reqVisible1.SendVectorToReduce(rec_visible_units2);

					// B5) Async ricezione Vrec
					reqVisibleRicezione2.ReceiveVector(rec_visible_units2);

					//Prodotto matriciale V' * W
					matrix_transpose_vector_multiplication(weights, rec_visible_units1, rec_hidden_units1);

					// B4) Wait invio Vrec
					reqVisible1.wait();

					// B5) Wait ricezione Vrec
					reqVisibleRicezione2.wait();



					// A6) Async invio H rec
					reqHidden1.SendVectorToReduce(rec_hidden_units1);

					// A7) Async ricezione H rec
					reqHiddenRicezione1.ReceiveVector(rec_hidden_units1);

					//Prodotto matriciale V' * W
					matrix_transpose_vector_multiplication(weights, rec_visible_units2, rec_hidden_units2);

					// A6) Wait invio H rec
					reqHidden1.wait();

					// A7) Wait ricezione H rec
					reqHiddenRicezione1.wait();





					// B6) Async invio H rec
					reqHidden1.SendVectorToReduce(rec_hidden_units2);

					// B7) Async ricezione H rec
					reqHiddenRicezione1.ReceiveVector(rec_hidden_units2);

					//gradiente
					//si calcolano i differenziali dei pesi
					set_gradient(diff_weights, visible_units1, hidden_units1, rec_visible_units1, rec_hidden_units1);

					// B6) Wait invio H rec
					reqHidden1.wait();

					// B7) Wait ricezione H rec
					reqHiddenRicezione1.wait();



					// A1) Async ricezione V
					const bool other_samples = current_index_sample != rbm_n_training_epocs * number_of_samples;
					if(other_samples)
						reqVisible1.ReceiveVector(visible_units1);

					//gradiente
					//si calcolano i differenziali dei pesi
					set_gradient(diff_weights, visible_units2, hidden_units2, rec_visible_units2, rec_hidden_units2);

					//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
					if(current_index_sample % rbm_size_minibatch == 0)
						update_parameters_weights(rbm_momentum, current_learning_rate, weights, diff_weights, number_of_samples);

				} //fine esempio
			} //fine epoca


			//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
			int n_last_samples = current_index_sample % rbm_size_minibatch;
			if(n_last_samples != 0)
				update_parameters_weights(rbm_momentum, current_learning_rate, weights, diff_weights, n_last_samples);

			//SALVATAGGIO NUOVI INPUT (non viene sfruttato il doppio canale come nel training della RBM)
			save_new_samples(reqVisible1,reqHidden1, weights,
					visible_units1, visible_units2, hidden_units1, hidden_units2);


			//contatore che memorizza il numero di rbm apprese
			trained_rbms++;
			save_parameters();

		}//fine layer

	}




	void node_cell_autoencoder::save_new_samples(
			MPReqManagerCell& reqVis, MPReqManagerCell& reqHid,
			matrix<float>& weights,
			my_vector<float>& visible_units1, my_vector<float>& visible_units2,
			my_vector<float>& hidden_units1, my_vector<float>& hidden_units2)
	{
		// 1) Async ricezione V
		//ReceiveVector(comms_for_vis, visible_units1, reqsVis1);
		reqVis.ReceiveVector(visible_units1);


		for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
		{
			// 1) Wait ricezione V
			reqVis.wait();

			visible_units2 = visible_units1; //copia valori in un altro buffer

			// 1) Async ricezione V
			if(current_index_sample != number_of_samples - 1)
				reqVis.ReceiveVector(visible_units1);

			//Calcolo
			matrix_transpose_vector_multiplication(weights, visible_units2, hidden_units1);

			// 3) Wait Invio H
			if(current_index_sample != 0)
				reqHid.wait();

			// 3) Async Invio H
			hidden_units2 = hidden_units1; //utilizzo un altro buffer

			reqHid.SendVectorToReduce(hidden_units1);
		}

		// 3) Wait Invio H
		reqHid.wait();
	}




	void node_cell_autoencoder::get_activation_output_layers(my_vector<my_vector<float>>& activation_layers, my_vector<my_vector<float>>& output_layers)
	{
		for(uint l = 0; l != activation_layers.size(); l++)
		{
			uint n_my_visible_units, n_my_hidden_units;
			get_my_visible_hidden_units(l,n_my_visible_units, n_my_hidden_units);

			activation_layers[l] = my_vector<float>(n_my_visible_units);
			output_layers[l] = my_vector<float>(n_my_hidden_units);
		}
	}


	void node_cell_autoencoder::forward_pass(my_vector<my_vector<float>>& activation_layers, my_vector<my_vector<float>>& output_layers)
	{
		//1. forward pass
		for(uint l = 0; l != number_of_final_layers - 1; l++){

			//comunicatori
			auto& comms_for_vis = acc_vis_comm_for_layer[l];
			auto& comms_for_hid = acc_hid_comm_for_layer[l];

			//bias, input e output vectors
			auto& weights = layers_weights[l];
			auto& input = activation_layers[l];
			auto& output = output_layers[l];


			//Ricezione da accumulatori
			MPI_Request vis_requests[comms_for_vis.size()];
			MPI_Request reqs_hid[comms_for_hid.size()];

			MPReqManagerCell reqVis{ vis_requests, &comms_for_vis};
			MPReqManagerCell reqHid{ reqs_hid, &comms_for_hid};


			reqVis.ReceiveVectorSync(input);

			//Calcolo matriciale H = V * W
			matrix_transpose_vector_multiplication(weights, input, output);

			//Invio ad accumulatori
			reqHid.SendVectorToReduce(output);

		} //fine forward
	}

	void node_cell_autoencoder::backward_pass(my_vector<my_vector<float>>& activation_layers,
			my_vector<my_vector<float>>& output_layers)
	{
		//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
		for(uint l = number_of_final_layers - 1; l != 0; l--){

			//comunicatori
			auto& comms_for_vis = acc_vis_comm_for_layer[l - 1];
			auto& comms_for_hid = acc_hid_comm_for_layer[l - 1];

			//si aggiornano i pesi tra l'output e l'input layer
			auto& weights_to_update = layers_weights[l - 1];

			auto& output_layer = output_layers[l - 1];
			auto& input_layer = activation_layers[l - 1];

			//vettori delta in input e calcolati
			auto deltas_to_accs = my_vector<float>(input_layer.size());
			auto deltas_from_accs = my_vector<float>(output_layer.size());

			//check misure
			assert(output_layer.size() == weights_to_update.get_cols());
			assert(input_layer.size() == weights_to_update.get_rows());


			//ricevi delta da accumulatori
			MPI_Request req_from_acc[comms_for_hid.size()];
			MPReqManagerCell reqFromAcc{req_from_acc,&comms_for_hid };

			reqFromAcc.ReceiveVectorSync(deltas_from_accs);

			//calcola delta pesati (li si moltiplicano per la matrice dei pesi)
			matrix_vector_multiplication(weights_to_update, deltas_from_accs, deltas_to_accs);

			//Invia delta pesati agli accumulatori (all'ultimo passo non è necessario)
			if(l > 1)
			{
				MPI_Request reqs_to_acc[comms_for_vis.size()];
				MPReqManagerCell reqToAcc{ reqs_to_acc, &comms_for_vis  };

				reqToAcc.SendVectorToReduce(deltas_to_accs);
			}

			//applico gradiente per la matrice dei pesi
			update_weights_fine_tuning(weights_to_update, deltas_from_accs, input_layer, fine_tuning_learning_rate);
		}
	}


	void node_cell_autoencoder::rollup_for_weights()
	{
		//Roll-up
		//una volta appresi i pesi, bisogna creare una rete di tipo feed forward
		//la rete feed forward dell'autoencoder possiede il doppio dei layer hidden,
		//ad eccezione del layer più piccolo che di fatto serve a memorizzare l'informazione in maniera più corta

		for(uint trained_layer = 0; trained_layer != number_of_rbm_to_learn; trained_layer++)
		{
			//memorizzo i pesi trasposti nel layer feed forward, attenzione agli indici
			const uint index_weights_dest = number_of_final_layers - trained_layer - 2;

			//si salva la trasposta dei pesi
			auto& layer_weights_source = layers_weights[trained_layer];
			auto& layer_weights_dest = layers_weights[index_weights_dest];

			layer_weights_dest = matrix<float>(layer_weights_source.get_cols(), layer_weights_source.get_rows());

			//todo assicurarsi che a livello globale la matrice dei pesi sia stata trasposta
			//grazie all'orientamento inverso dovrebbe essere così
			transpose_matrix(layer_weights_source, layer_weights_dest);
		}
	}

	void node_cell_autoencoder::fine_tuning()
	{
		rollup_for_weights();

		//INIZIO FINE TUNING
		//si riserva lo spazio necessario per l'attivazione di ogni layer
		// e per l'output restituito ad ogni layer
		my_vector<my_vector<float>> activation_layers(orientation_grid.size());
		my_vector<my_vector<float>> output_layers(orientation_grid.size());
		get_activation_output_layers(activation_layers, output_layers);

		//per ogni epoca e ogni suo esempio...
		for(uint current_index_sample = 0;
				current_index_sample != number_of_samples * fine_tuning_n_training_epocs;
				current_index_sample++)
		{
			//1. forward pass
			forward_pass(activation_layers, output_layers);

			//2. backward pass
			backward_pass(activation_layers, output_layers);
		} //fine esempi



		//allenamento concluso
		fine_tuning_finished = true;
		save_parameters();


	}


	my_vector<float> node_cell_autoencoder::reconstruct(){

		my_vector<my_vector<float>> activation_layers(orientation_grid.size());
		my_vector<my_vector<float>> output_layers(orientation_grid.size());
		get_activation_output_layers(activation_layers, output_layers);

		//1. forward pass
		forward_pass(activation_layers, output_layers);

		return output_layers[output_layers.size() - 1]; //dummy
	}


	string node_cell_autoencoder::get_path_file(){

		return folder_parameters_path + "paral_cell_"+ std::to_string(row_number) + "x" + std::to_string(col_number) + ".txt";
	}


	void node_cell_autoencoder::save_parameters()
	{

		string path_file = get_path_file();

		// Create an input filestream
		std::ofstream myFile(path_file);

		// Make sure the file is open
		if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

		//salvataggio di pesi, bias
		uint layer_number;
		for(layer_number = 0; layer_number != trained_rbms; layer_number++)
		{
			auto& weights = layers_weights[layer_number];

			myFile << "_rbm_" << weights.get_rows() << "x" << weights.get_cols() << "__,";
			for(uint i = 0; i != weights.size(); i++)
				myFile << fixed << setprecision(F_PREC) << weights[i] << ",";

			myFile << endl;
		}


		if(fine_tuning_finished)
		{
			while(layer_number < number_of_final_layers - 1)
			{
				auto& weights = layers_weights[layer_number];

				myFile << "_rec_" << weights.get_rows() << "x" << weights.get_cols() << "__,";
				for(uint i = 0; i != weights.size(); i++)
						myFile << fixed << setprecision(F_PREC) << weights[i] << ",";

				myFile << endl;

				layer_number++;
			}
		}

		myFile.close();

	}

	void node_cell_autoencoder::load_parameters(){

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
		uint n_my_visible_units;
		uint n_my_hidden_units;

		matrix<float> *current_weights;

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
			layers_weights[trained_rbms] = matrix<float>(n_my_visible_units, n_my_hidden_units);
			current_weights = &layers_weights[trained_rbms];

			//questa variabile memorizza il numero di rbm apprese
			trained_rbms++;


			// Create a stringstream of the current line

			std::stringstream ss(line);

			//riga dei pesi
			ss.ignore(100, ',');

			for(uint i = 0; i != n_my_visible_units; i++)
				for(uint j = 0; j != n_my_hidden_units; j++){
					if(ss.peek() == ',') ss.ignore();
					ss >> current_weights->at(i, j);
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
				layers_weights[current_layer] = matrix<float>(n_my_visible_units, n_my_hidden_units);
				current_weights = &layers_weights[current_layer];

				current_layer++;

				// Create a stringstream of the current line
				std::stringstream ss(line);

				//riga dei pesi
				ss.ignore(100, ',');
				for(uint i = 0; i != n_my_visible_units; i++)
					for(uint j = 0; j != n_my_hidden_units; j++){
					   if(ss.peek() == ',') ss.ignore();
					   ss >>  current_weights->at(i, j);
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
