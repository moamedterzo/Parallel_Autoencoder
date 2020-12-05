/*
 * node_single_autoencoder.cpp
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#include "node_single_autoencoder.h"

#include <sstream>
#include <fstream>
#include <cstring>
#include <iomanip>

using namespace std;


namespace parallel_autoencoder
{

	//imposta gradienti per pesi e bias
	inline void set_gradient(my_vector<float>& visible_units, my_vector<float>& hidden_units,
			my_vector<float>& rec_visible_units, my_vector<float>& rec_hidden_units,
			my_vector<float>& diff_visible_biases, my_vector<float>& diff_hidden_biases,
			matrix<float>& diff_weights)
	{
		for(uint i = 0; i != visible_units.size(); i++)
		{
			diff_visible_biases[i] += visible_units[i] - rec_visible_units[i];

			for(uint j = 0; j != hidden_units.size(); j++){
				diff_weights.at(i, j) +=  visible_units[i] * hidden_units[j]  //fattore positivo
										  - rec_visible_units[i] * rec_hidden_units[j]; //fattore negativo
			}
		}

		//dei bias nascosti
		for(uint j = 0; j != hidden_units.size(); j++)
			diff_hidden_biases[j] +=  hidden_units[j] - rec_hidden_units[j];
	}


	node_single_autoencoder::node_single_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
				uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode, bool _reduce_io,
				std::ostream& _oslog,
				samples_manager& _smp_manager)
		: node_autoencoder(_layers_size, _generator, 0, 0, 0,
							rbm_n_epochs, finetuning_n_epochs, batch_mode, _reduce_io, _oslog, 0)
	{
		smp_manager = _smp_manager;

		image_path_folder = string(smp_manager.path_folder);

		 //Per N layer bisogna apprendere N-1 matrici di pesi e N-1 vettori di bias
		layers_weights = my_vector<matrix<float>>(number_of_final_layers - 1);
		layer_biases = my_vector<my_vector<float>>(number_of_final_layers - 1);

		//ottengo il numero di elementi del dataset
		number_of_samples = smp_manager.get_number_samples();

		//Al momento il numero di esempi deve essere necessariamente pari
		if(number_of_samples % 2 != 0)
			std::cout << "Number of samples must be even\nPOSSIBLE ERRORS\n";
	}


	CommandType node_single_autoencoder::wait_for_command()
	{
		CommandType command;

		std::cout << "\n\nHello, I'm the master, what do you want to do?\n";

		std::cout << "1 to train a rbm\n";
		if(fine_tuning_finished) std::cout << "2 to save parameters\n";
		std::cout << "3 to load parameters\n";
		if(fine_tuning_finished) std::cout << "5 to reconstruct images\n";
		std::cout << "-1 to exit\n";
		std::cout << "-22 to delete parameters file\n";

		//lettura input
		int res;
		std::cin >> res;

		//determinazione comando e dati aggiuntivi
		char char_path_file[MAX_FOLDER_PARS_LENGTH];
		if(res == 1)
		{
			command = CommandType::train;
		}
		else if(res == 2)
		{
			command = CommandType::save_pars;

			std::cout << "If you want to save parameters, you can specify a custom folder path (or '.' as default):\n";
			std::cin >> char_path_file;
		}
		else if(res == 3)
		{
			command = CommandType::load_pars;

			std::cout << "If you want to load parameters, you can specify a custom folder path (or '.' as default):\n";
			std::cin >> char_path_file;
		}
		else if(res == -22)
			command = CommandType::delete_pars_file;
		else if(res == 5)
			command = CommandType::reconstruct_image;
		else if(res == -1)
			command = CommandType::exit;
		else
			command = CommandType::retry;


		if(command == CommandType::load_pars || command == CommandType::save_pars)
		{
			//cartella che contiene i parametri
			if(strcmp(char_path_file, ".") != 0)
				folder_parameters_path = string(char_path_file);
		}


		return command;
	}

	void node_single_autoencoder::train_rbm()
	{
		//il numero del minibatch non può essere più grande del numero di esempi
		rbm_size_minibatch = std::min(rbm_size_minibatch, number_of_samples);

		//percorso della cartella che contiene le immagini iniziali
		string image_path_folder = string(smp_manager.path_folder);

		//Per ciascun layer...
		//se sono stati già apprese delle rbm, si passa direttamente alla successiva da imparare
		for(uint layer_number = trained_rbms; layer_number != number_of_rbm_to_learn; layer_number++)
		{
			//numero di unità gestite
			uint n_visible_units = layers_size[layer_number];
			uint n_hidden_units = layers_size[layer_number + 1];

			const char *sample_extension = layer_number == 0 ?  default_extension.c_str() : ".txt";

			std::cout << "-- Learning layer number: " << layer_number
					<< ", visible units: " << n_visible_units
					<< ", hidden units: " << n_hidden_units	<< " --\n";

			//RBM
			const bool first_layer = layer_number == 0;
			const uint index_reverse_layer = number_of_final_layers - layer_number - 2;

			auto& weights = layers_weights[layer_number];
			auto& hidden_biases = layer_biases[layer_number];
			auto& visible_biases =  layer_biases[index_reverse_layer];
			{
				//la matrice dei pesi per il layer in questione,
				//possiede grandezza VxH (unità visibili per unità nascoste)
				//si riserva lo spazio necessario
				weights = matrix<float>(n_visible_units, n_hidden_units, 0.0);

				//inizializzazione pesi
				initialize_weight_matrix(weights, rbm_initial_weights_mean, rbm_initial_weights_variance, generator);

				//inizializzazione bias
				visible_biases = my_vector<float>(n_visible_units, rbm_initial_biases_value);
				hidden_biases = my_vector<float>(n_hidden_units, rbm_initial_biases_value);

				//layers visibili e nascosti, ricostruiti e non
				my_vector<float> visible_units(n_visible_units, 0.0);
				my_vector<float> hidden_units(n_hidden_units, 0.0);
				my_vector<float> rec_visible_units(n_visible_units, 0.0);
				my_vector<float> rec_hidden_units(n_hidden_units, 0.0);

				//gradienti calcolati per pesi e bias
				matrix<float> diff_weights(n_visible_units, n_hidden_units, 0.0);
				my_vector<float> diff_visible_biases(n_visible_units, 0.0);
				my_vector<float> diff_hidden_biases(n_hidden_units, 0.0);

				//si avvia il processo di apprendimento per diverse epoche
				ulong current_index_sample = 0;
				float learning_rate;

				for(uint epoch = 0; epoch != rbm_n_training_epocs; epoch++){

					learning_rate = get_learning_rate_rbm(epoch, layer_number);

					if(epoch % 1 == 0)
						std::cout << "Training epoch: " << epoch << "\n";

					smp_manager.restart();

					//per ciascun esempio...
					while(smp_manager.get_next_sample(visible_units, sample_extension))
					{
						current_index_sample++;

						//CONTRASTIVE DIVERGENCE

						//1. Effettuare sampling dell'hidden layer
						matrix_transpose_vector_multiplication(weights, visible_units, hidden_units);
						sample_hidden_units(hidden_units, hidden_biases, generator);

						//2. Ricostruire layer visibile
						matrix_vector_multiplication(weights, hidden_units, rec_visible_units);
						reconstruct_visible_units(rec_visible_units, visible_biases, first_layer, generator);

						//3. si ottiene il vettore hidden partendo dalle unità visibili ricostruite
						matrix_transpose_vector_multiplication(weights, rec_visible_units, rec_hidden_units);
						reconstruct_hidden_units(rec_hidden_units, hidden_biases, first_layer, generator);

						//4. si calcolano i differenziali dei pesi e bias visibili
						set_gradient(visible_units, hidden_units, rec_visible_units, rec_hidden_units,
								diff_visible_biases, diff_hidden_biases, diff_weights);

						//se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
						if(current_index_sample % rbm_size_minibatch == 0)
						{
							update_biases_rbm(rbm_momentum, learning_rate, hidden_biases, visible_biases,
									diff_visible_biases, diff_hidden_biases, rbm_size_minibatch);

							update_weights_rbm(rbm_momentum, learning_rate, weights, diff_weights, rbm_size_minibatch);
						}
					}
				}

				//se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
				int n_last_samples = current_index_sample % rbm_size_minibatch;
				if(n_last_samples != 0)
				{
					update_biases_rbm(rbm_momentum, learning_rate, hidden_biases, visible_biases,
							diff_visible_biases, diff_hidden_biases, n_last_samples);

					update_weights_rbm(rbm_momentum, learning_rate, weights, diff_weights, n_last_samples);
				}
			}


			std::cout<< "New RBM trained\n";

			//SALVATAGGIO IMMAGINI
			save_new_samples(layer_number, n_visible_units, n_hidden_units, sample_extension, hidden_biases, weights);

			//contatore che memorizza il numero di rbm apprese
			trained_rbms++;

			if(!reduce_io)
				save_parameters();
		}
	}

	void node_single_autoencoder::save_new_samples(const uint layer_number,const uint n_visible_units,const uint n_hidden_units,
			const char *sample_extension,
			my_vector<float>& hidden_biases, matrix<float>& weights)
	{
		//si deve salvare sul disco i risultati di attivazione del layer successivo
		//essi saranno utilizzati come input per la prossima fare di training
		string new_image_path_folder = string(image_path_folder + "/" + std::to_string(layer_number));

		std::cout << "Saving next inputs for RBM in folder '" << new_image_path_folder << "'\n";

		//vettori
		string sample_filename;
		my_vector<float> input_samples(n_visible_units);
		my_vector<float> output_samples(n_hidden_units);

		smp_manager.restart();
		while(smp_manager.get_next_sample(input_samples, sample_extension, &sample_filename)){

			//si ottengono i valori di attivazione dalla RBM
			matrix_transpose_vector_multiplication(weights, input_samples, output_samples);
			apply_sigmoid_to_layer(output_samples, hidden_biases, false);

			//si salva su file
			smp_manager.save_sample(output_samples, false, new_image_path_folder, sample_filename + ".txt"); //dati in formato testuale

			if(!reduce_io)
				smp_manager.save_sample(output_samples, true, new_image_path_folder, sample_filename +  default_extension); //dati in formato immagine
		}

		//in maniera del tutto trasparente si utilizzerà questo nuovo percorso per ottenere i dati in input
		smp_manager.path_folder = new_image_path_folder;
	}




	void node_single_autoencoder::rollup_for_weights()
	{
		for(uint trained_layer = 0; trained_layer != number_of_rbm_to_learn; trained_layer++)
		{
			//memorizzo i pesi trasposti nel layer feed forward
			const uint index_weights_dest = number_of_final_layers - trained_layer - 2;

			//si salva la trasposta dei pesi
			auto& layer_weights_source = layers_weights[trained_layer];
			auto& layer_weights_dest = layers_weights[index_weights_dest];

			layer_weights_dest = matrix<float>(layer_weights_source.get_cols(), layer_weights_source.get_rows());

			transpose_matrix(layer_weights_source, layer_weights_dest);
		}
	}

	void node_single_autoencoder::forward_pass(my_vector<my_vector<float>>& activation_layers)
	{
		const uint central_layer = number_of_final_layers / 2 - 1;

		for(uint l = 1; l != number_of_final_layers; l++){

			auto& weights = layers_weights[l - 1];
			auto& biases = layer_biases[l - 1];
			auto& input = activation_layers[l - 1];
			auto& activation_layer = activation_layers[l];

			//l'attivazione del layer successivo è data dai pesi e dal bias
			matrix_transpose_vector_multiplication(weights, input, activation_layer);

			//si applica la funzione sigmoide
			//se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
			//per ottenere un valore binario
			apply_sigmoid_to_layer(activation_layer, biases, l == central_layer);
		}
	}

	void node_single_autoencoder::backward_pass(my_vector<my_vector<float>>& activation_layers)
	{
		//si va dall'ultimo layer al penultimo (quello di input non viene considerato)
		my_vector<float> former_deltas;
		my_vector<float> current_deltas;

		for(uint l = number_of_final_layers - 1; l != 0; l--){

			//si aggiornano i pesi tra l'output e l'input layer
			auto& weights_to_update = layers_weights[l - 1];
			auto& biases_to_update = layer_biases[l - 1];

			auto& output_layer = activation_layers[l];
			auto& input_layer = activation_layers[l - 1];

			//check sizes
			assert(output_layer.size() == biases_to_update.size());
			assert(output_layer.size() == weights_to_update.get_cols());
			assert(input_layer.size() == weights_to_update.get_rows());

			//si calcoleranno i delta per il layer corrente
			if(l == number_of_final_layers - 1)
			{
				//layer di output
				current_deltas = my_vector<float>(output_layer.size(), 0.0);

				//calcolo dei delta per il layer di output
				deltas_for_output_layer(output_layer, activation_layers[0], current_deltas);
			}
			else
			{
				//layer nascosto

				//si memorizzano i delta del passo precedente
				former_deltas = my_vector<float>(current_deltas);
				current_deltas = my_vector<float>(output_layer.size(), 0.0);

				//si vanno a prendere i pesi tra il layer di output e quello a lui successivo
				auto& weights_for_deltas = layers_weights[l];

				//il delta per il nodo j-esimo è dato dalla somma pesata dei delta dei nodi del layer successivo
				matrix_vector_multiplication(weights_for_deltas, former_deltas, current_deltas);
			}

			//applico gradiente per la matrice dei pesi
			update_weights_fine_tuning(weights_to_update, current_deltas, input_layer, fine_tuning_learning_rate);

			//si applica il gradiente per i bias
			update_biases_fine_tuning(biases_to_update, current_deltas, fine_tuning_learning_rate);
		}
	}

	my_vector<my_vector<float>> node_single_autoencoder::get_activation_layers()
	{
		my_vector<my_vector<float>> activation_layers(number_of_final_layers);

		for(uint l = 0; l != activation_layers.size(); l++)
			//la grandezza è memorizzata nel vettore layers_size
			activation_layers[l] = my_vector<float>(layers_size[l]);

		return activation_layers;
	}



	void node_single_autoencoder::fine_tuning()
	{
		 //Roll-up
		rollup_for_weights();


		//Fine tuning
		std::cout << "\n\nFINE TUNING\n";

		//si riserva lo spazio necessario per l'attivazione di ogni layer
		//e per i vettori che conterranno i valori delta per la back propagation
		auto activation_layers = get_activation_layers();

		my_vector<my_vector<float>> delta_layers(number_of_final_layers - 1);
		for(uint l = 0; l != delta_layers.size(); l++)
			delta_layers[l] = my_vector<float>(layers_size[l + 1]);


		//si passa alle immagini iniziali
		smp_manager.path_folder = image_path_folder;

		//per ogni epoca...
		for(uint epoch = 0; epoch != fine_tuning_n_training_epocs; epoch++)
		{
			smp_manager.restart();

			std::cout << "Training epoch: " << epoch << "\n";

			//per ciascun esempio...
			while(smp_manager.get_next_sample(activation_layers[0],  default_extension.c_str() )){

				//1. forward pass
				forward_pass(activation_layers);

				//2. backward pass
				backward_pass(activation_layers);
			}
		}

		//allenamento concluso
		fine_tuning_finished = true;

		if(!reduce_io)
			save_parameters();
	}




	void node_single_autoencoder::reconstruct()
	{

		//si prende l'immagine dal manager
		smp_manager.path_folder = image_path_folder;
		smp_manager.restart();
		string file_name;

		//vettori per l'attivazione dei layer
		auto activation_layers = get_activation_layers();
		my_vector<float>& output = activation_layers[activation_layers.size() - 1];
		my_vector<float>& input = activation_layers[0];

		//errore medio quadratico
		float mean_root_squared_error = 0;

		for(uint i = 0; i != number_of_samples; i++)
		{
			//0. Prelevo esempio da IO
			smp_manager.get_next_sample(input,  default_extension.c_str(), &file_name);

			//1. forward pass
			forward_pass(activation_layers);


			if(batch_mode)
			{
				//si salvano i risultati su file
				string new_image_path_folder = string(image_path_folder + "/output_rec" );
				smp_manager.save_sample(output, true, new_image_path_folder, file_name + default_extension);
			}
			else
			{
				//si mostra a video il risultato
				std::cout << "Showing original sample: '" <<  smp_manager.path_folder << "/" << file_name << "'\n";
				smp_manager.show_sample(activation_layers[0]);

				std::cout << "Showing reconstructed sample:\n";
				smp_manager.show_sample(output);

			}

			mean_root_squared_error += root_squared_error(input, output);
		}

		std::cout << "Mean root squared error: " << mean_root_squared_error/number_of_samples << "\n";
	}

	string node_single_autoencoder::get_path_file(){

		return folder_parameters_path + "single_master.txt";
	}

	void node_single_autoencoder::save_parameters()
	{
		string path_file = get_path_file();

		std::cout << "Saving autoencoder parameters to '" + path_file + "'\n";

		// Create an input filestream
		std::ofstream myFile(path_file);

		// Make sure the file is open
		if(!myFile.is_open()) cout << "Could not open file: " + path_file << "\n";

		//salvataggio di pesi, bias
		uint layer_number;
		for(layer_number = 0; layer_number != trained_rbms; layer_number++)
		{
			const uint index_reverse_layer = number_of_final_layers - layer_number - 2;

			auto& weights = layers_weights[layer_number];
			auto& hidden_biases = layer_biases[layer_number];
			auto& visible_biases =  layer_biases[index_reverse_layer];


			myFile << "_rbm_" << weights.get_rows() << "x" << weights.get_cols() << "__,";
			for(uint i = 0; i < weights.size(); i++)
				  myFile << fixed << setprecision(F_PREC) << weights[i] << ",";
			myFile << endl;

			myFile << "_hidden_" << hidden_biases.size() << "__,";
			for(uint i = 0; i < hidden_biases.size(); i++)
				myFile << fixed << setprecision(F_PREC) <<hidden_biases[i] << ",";
			myFile << endl;

			myFile << "_visible_" << visible_biases.size() << "__,";
			for(uint i = 0; i < visible_biases.size(); i++)
				myFile << fixed << setprecision(F_PREC) <<visible_biases[i] << ",";
			myFile << endl;
		}

		if(fine_tuning_finished)
		{
			while(layer_number != number_of_final_layers - 1)
			{
				auto& weights = layers_weights[layer_number];
				auto& hidden_biases = layer_biases[layer_number];

				myFile << "_rec_" <<  weights.get_rows() << "x" << weights.get_cols()  << "__,";
				for(uint i = 0; i < weights.size(); i++)
					 myFile << fixed << setprecision(F_PREC) << weights[i] << ",";
				myFile << endl;

				myFile << "_rec_" << hidden_biases.size() << "__,";
				for(uint i = 0; i < hidden_biases.size(); i++)
					  myFile << fixed << setprecision(F_PREC) <<hidden_biases[i] << ",";
				myFile << endl;

				layer_number++;
			}
		}

		myFile.close();
	}

	void node_single_autoencoder::load_parameters()
	{
		string path_file = get_path_file();

		 std::cout << "Getting autoencoder parameters from '" + path_file + "'\n";

		fine_tuning_finished = false;
		trained_rbms = 0;

		// Create an input filestream
		std::ifstream myFile(path_file);

		// Make sure the file is open
		if(!myFile.is_open()) cout << "Could not open file: " + path_file << "\n";

		// Helper vars
		std::string line;

		//variabili che fanno riferimento al layer nel quale si salveranno i parametri
		uint n_visible_units {};
		uint n_hidden_units {};

		matrix<float> *current_weights;
		my_vector<float> *current_hidden_biases;
		my_vector<float> *current_visible_biases;

		uint current_row_file = 0;

		//Si leggono le linee che contengono i pesi delle rbm apprese
		bool other_lines = false;
		while(std::getline(myFile, line))
		{
			if(current_row_file % 3 == 0)
			{
				//se abbiamo letto tutti i pesi delle rbm si esce da questo ciclo
				if(trained_rbms == number_of_rbm_to_learn){
					//se ci sono altre linee, vuol dire che si possiedono i parametri dei layer di ricostruzione
					other_lines = true;
					break;
				}

				const uint index_reverse_layer = number_of_final_layers - trained_rbms - 2;

				n_visible_units = layers_size[trained_rbms];
				n_hidden_units = layers_size[trained_rbms + 1];

				layers_weights[trained_rbms] = matrix<float>(n_visible_units, n_hidden_units);
				layer_biases[trained_rbms] = my_vector<float>(n_hidden_units);
				layer_biases[index_reverse_layer] = my_vector<float>(n_visible_units);

				current_weights = &layers_weights[trained_rbms];
				current_hidden_biases = &layer_biases[trained_rbms];
				current_visible_biases = &layer_biases[index_reverse_layer];

				//questa variabile memorizza il numero di rbm apprese
				trained_rbms++;
			}

			// Create a stringstream of the current line
			std::stringstream ss(line);

			//in base alla riga si aggiornano i relativi parametri
			if(current_row_file % 3 == 0){

				//riga dei pesi
				ss.ignore(100, ',');
				for(uint i = 0; i != n_visible_units; i++)
					for(uint j = 0; j != n_hidden_units; j++){
					   if(ss.peek() == ',') ss.ignore();
					   ss >>  current_weights->at(i, j);
					}

			}
			else if (current_row_file % 3 == 1){

				//riga dei bias nascosti
				ss.ignore(100, ',');
				for(uint j = 0; j != n_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->operator [](j);
					}
			}
			else if (current_row_file % 3 == 2){

				//riga dei bias visibili
				ss.ignore(100, ',');
				for(uint i = 0; i != n_visible_units; i++) {
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
			current_row_file = 0;

			//indice del layer contenente pesi o bias
			uint current_layer = (number_of_final_layers - 1) / 2;
			do
			{
				if(current_row_file % 2 == 0)
				{
					n_visible_units = layers_size[current_layer];
					n_hidden_units = layers_size[current_layer + 1];

					layers_weights[current_layer] = matrix<float>(n_visible_units, n_hidden_units);

					//già inizializzati
					current_weights = &layers_weights[current_layer];
					current_hidden_biases = &layer_biases[current_layer];
					//c'è un solo layer per i bias da aggiornare

					current_layer++;

					//check sulle misure
					assert(n_hidden_units == current_hidden_biases->size());
				}

				// Create a stringstream of the current line
				std::stringstream ss(line);

				//in base alla riga si aggiornano i relativi parametri
				if(current_row_file % 2 == 0){

					//riga dei pesi
					ss.ignore(100, ',');
					for(uint i = 0; i != n_visible_units; i++)
						for(uint j = 0; j != n_hidden_units; j++){
						   if(ss.peek() == ',') ss.ignore();
						   ss >>  current_weights->at(i, j);
						}
				}
				else if (current_row_file % 2 == 1){

					//riga dei bias
					ss.ignore(100, ',');
					for(uint j = 0; j != n_hidden_units; j++){
							if(ss.peek() == ',') ss.ignore();
							ss >> current_hidden_biases->operator [](j);
						}
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


}


