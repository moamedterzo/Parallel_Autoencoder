/*
 * node_master_autoencoder.cpp
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#include "node_master_autoencoder.h"

#include <sstream>
#include <fstream>
#include <string.h>

using namespace std;


namespace parallel_autoencoder
{


	void node_master_autoencoder::scatter_vector(const my_vector<float>& vec, const int send_counts[], const int displs[], MPI_Request *reqSend)
	{
		MPI_Iscatterv(vec.data(), send_counts, displs, mpi_datatype_tosend,
				    MPI_IN_PLACE, 0,  mpi_datatype_tosend,
					0, master_accs_comm, reqSend);
	}

	void node_master_autoencoder::gather_vector(const my_vector<float>& vec, const int receive_counts[], const int displs[], MPI_Request *reqRecv)
	{
		MPI_Igatherv(MPI_IN_PLACE, 0, mpi_datatype_tosend,
				vec.data(), receive_counts, displs, mpi_datatype_tosend,
				0, master_accs_comm,  reqRecv);
	}

	void node_master_autoencoder::scatter_vector_sync(const my_vector<float>& vec, const int send_counts[], const int displs[])
	{
		MPI_Scatterv(vec.data(), send_counts, displs, mpi_datatype_tosend,
					nullptr, 0,  mpi_datatype_tosend,
					0, master_accs_comm);
	}

	void node_master_autoencoder::gather_vector_sync(const my_vector<float>& vec, const int receive_counts[], const int displs[])
	{
		MPI_Gatherv(MPI_IN_PLACE, 0, mpi_datatype_tosend,
				vec.data(), receive_counts, displs, mpi_datatype_tosend,
				0, master_accs_comm);
	}




	void node_master_autoencoder::get_scatter_parts(int counts[], int displacements[], const int n_total_units)
	{
		//root (master) non riceve nulla
		counts[0] = displacements[0] = 0;

		uint n_units_for_acc = 0;
		for(uint k = 0; k != total_accumulators; k++){

			//l'accumulatore k è posizionato alla posizione k + 1 (a causa dell'indice 0 che rappresenta il root)
			displacements[k + 1] = displacements[k] + n_units_for_acc;

			//calcolo elementi da inviare/ricevere per l'accumulatore
			n_units_for_acc = get_units_for_node(n_total_units, total_accumulators, k);
			counts[k + 1] = n_units_for_acc;
		}
	}


	node_master_autoencoder::node_master_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
				uint _total_accumulators, uint _grid_row, uint _grid_col,
				uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode, bool _reduce_io,
				std::ostream& _oslog, int _mpi_rank,
				MPI_Comm& _master_accs_comm,
				samples_manager& _smp_manager)
		: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col,rbm_n_epochs, finetuning_n_epochs, batch_mode,_reduce_io, _oslog, _mpi_rank)
	{
		smp_manager = _smp_manager;
		master_accs_comm = _master_accs_comm;

		image_path_folder = string(smp_manager.path_folder);

		//indico il numero di elementi del dataset
		number_of_samples = smp_manager.get_number_samples();

		//Al momento il numero di esempi deve essere necessariamente pari
		if(number_of_samples % 2 != 0)
			std::cout << "Number of samples must be even\nPOSSIBLE ERRORS\n";
	}


	CommandType node_master_autoencoder::wait_for_command()
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
			cin >> char_path_file;
		}
		else if(res == 3)
		{
			command = CommandType::load_pars;

			std::cout << "If you want to load parameters, you can specify a custom folder path (or '.' as default):\n";
			cin >> char_path_file;
		}
		else if(res == -22)
			command = CommandType::delete_pars_file;
		else if(res == 5)
			command = CommandType::reconstruct_image;
		else if(res == -1)
			command = CommandType::exit;


		//alert other node
		MPI_Bcast(&command,1, MPI_INT, 0, MPI_COMM_WORLD);


		//in base al comando si inviano dati aggiuntivi
		if(command == CommandType::load_pars || command == CommandType::save_pars)
		{
			//invio cartella che contiene i parametri
			if(strcmp(char_path_file, ".") != 0)
				folder_parameters_path = string(char_path_file);

			MPI_Bcast(char_path_file, MAX_FOLDER_PARS_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
		}


		return command;
	}





	void node_master_autoencoder::train_rbm()
	{
		//invio numero di esempi
		MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);

		//il numero del minibatch non può essere più grande del numero di esempi
		rbm_size_minibatch = std::min(rbm_size_minibatch, number_of_samples);

		//Per ciascun layer...
		//se sono stati già apprese delle rbm, si passa direttamente alla successiva da imparare
		for(uint layer_number = trained_rbms; layer_number != number_of_rbm_to_learn; layer_number++)
		{
			const uint n_visible_units = layers_size[layer_number];
			const uint n_hidden_units = layers_size[layer_number + 1];
			const char *sample_extension = layer_number == 0 ? default_extension.c_str() : ".txt";

			//si ottengono displacements per gli accumulatori
			int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
			get_scatter_parts(send_counts, send_displacements, n_visible_units);

			for(auto a : send_displacements)
				std::cout << "Displacement: " + to_string(a) + "\n";
			for(auto a : send_counts)
				std::cout << "send_counts: " + to_string(a) + "\n";

			std::cout << "-- Learning layer number: " << layer_number
					<< ", visible units: " << n_visible_units
					<< ", hidden units: " << n_hidden_units << " --\n";

			//gestori richieste
			MPI_Request reqSend;
			my_vector<float> visible_units(n_visible_units);
			my_vector<float> visible_units_send_buffer(n_visible_units);

			//si avvia il processo di apprendimento per diverse epoche
			for(uint epoch = 0; epoch != rbm_n_training_epocs; epoch++){

				//si riavvia l'ottenimento dei samples
				smp_manager.restart();

				MPI_Status ss;

				//leggo da file system il prossimo esempio mentre invio l'esempio precedente
				for(uint current_sample = 0; current_sample != number_of_samples; current_sample++)
				{
					//lettura file system
					smp_manager.get_next_sample(visible_units, sample_extension);

					//attendo completamento dell'invio precedente per poter inviare il prossimo vettore
					if(current_sample != 0)
					{
						//MPI_Wait(&reqSend, &ss);
						//print_ssa(&ss);
					}

					visible_units_send_buffer = visible_units;
					//scatter_vector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);
					scatter_vector_sync(visible_units_send_buffer, send_counts, send_displacements);
				}

				//si conclude l'ultimo invio effettuato
				//MPI_Wait(&reqSend, &ss);
				//print_ssa(&ss);
			}


			std::cout<< "RBM trained (master)\n";

			//SALVATAGGIO NUOVI INPUT
			save_new_samples(layer_number, n_visible_units, n_hidden_units, sample_extension,
					visible_units, visible_units_send_buffer);

			//contatore che memorizza il numero di rbm apprese
			trained_rbms++;

			if(!reduce_io)
				save_parameters();
		}

	}


	void node_master_autoencoder::save_new_samples(const uint layer_number,const uint n_visible_units,const uint n_hidden_units,
			const char *sample_extension,
			my_vector<float>& visible_units, my_vector<float>& visible_units_send_buffer)
	{
		//si devono salvare sul disco i risultati di attivazione del layer successivo
		//essi saranno utilizzati come input per la prossima fare di training
		string new_image_path_folder = string(image_path_folder + "/" + std::to_string(layer_number));
		std::cout << "Saving result for the next RBM in folder '"	<< new_image_path_folder << "'\n";

		//nome del file dove salvare ciascun esempio
		string sample_filename, sample_filename_prec;

		//risultato dell'operazione
		my_vector<float> output_samples(n_hidden_units, 0.0);

		//si ottengono displacements per gli accumulatori
		int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
		get_scatter_parts(send_counts, send_displacements, n_visible_units);

		int receive_counts[1 + total_accumulators], receive_displacements[1 + total_accumulators];
		get_scatter_parts(receive_counts, receive_displacements, n_hidden_units);

		//si fa in modo di ottimizzare il processo di scambio dei dati inviando e ricevendo contemporaneamente alle operazioni di I/O
		MPI_Request reqSend, reqRecv;

		//Si inizia a rileggere ciascun input e inviarlo
		smp_manager.restart();
		for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
		{
			//lettura esempio da file
			smp_manager.get_next_sample(visible_units, sample_extension, &sample_filename);

			if(current_index_sample != 0)
			{
				//attendo invio esempio precedente
				MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
			}

			//invio esempio (copia valori nel buffer)
			visible_units_send_buffer = visible_units;
			scatter_vector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);


			if(current_index_sample != 0)
			{
				//attesa gather
				gather_vector(output_samples, receive_counts, receive_displacements,&reqRecv);
				MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

				//si salva su file il vettore ottenuto
				 //dati in formato testuale utilizzati dalla RBM
				smp_manager.save_sample(output_samples, false, new_image_path_folder, sample_filename_prec + ".txt");

				//dati in formato immagine (non utilizzati dalla RBM)
				if(!reduce_io)
					smp_manager.save_sample(output_samples, true, new_image_path_folder, sample_filename_prec+  default_extension );
			}

			//dato che vengono letti prima due input e poi si inizia a salvare, è necessario memorizzare il nome del file precedente
			sample_filename_prec = sample_filename;
		}

		//attesa gather
		gather_vector(output_samples, receive_counts, receive_displacements,&reqRecv);
		MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

		//si salva su file il vettore Hidden ottenuto
		smp_manager.save_sample(output_samples, false, new_image_path_folder, sample_filename_prec + ".txt");
		if(!reduce_io)
			smp_manager.save_sample(output_samples, true, new_image_path_folder, sample_filename_prec+  default_extension);


		//si utilizzerà questo nuovo percorso per ottenere i dati in input
		smp_manager.path_folder = new_image_path_folder;
	}




	void node_master_autoencoder::fine_tuning()
	{
		std::cout << "Beginning Fine-tuning...\n" ;

		 //si passa nuovamente alle immagini iniziali
		smp_manager.path_folder = image_path_folder;

		//vettore delle unità visibili
		const uint n_visible_units = layers_size[0];
		auto visible_units = my_vector<float>(n_visible_units);
		auto visible_units_send_buffer = my_vector<float>(n_visible_units);

		//si ottengono displacements per gli accumulatori
		MPI_Request reqSend;
		int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
		get_scatter_parts(send_counts, send_displacements, n_visible_units);

		//per ogni epoca...
		for(uint epoch = 0; epoch != fine_tuning_n_training_epocs; epoch++)
		{
			smp_manager.restart();

			//mentre si invia un esempio si preleva il prossimo da file system
			for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
			{
				//lettura esempio da file
				smp_manager.get_next_sample(visible_units,  default_extension.c_str());

				if(current_index_sample != 0)
				{
					//attendo invio esempio precedente
					MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
				}

				//invio esempio (copia valori nel buffer)
				visible_units_send_buffer = visible_units;
				scatter_vector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);
			}

			//attendo invio esempio precedente
			MPI_Wait(&reqSend, MPI_STATUS_IGNORE);

		} //fine epoca


		//allenamento concluso
		fine_tuning_finished = true;

		if(!reduce_io)
			save_parameters();

		std::cout << "Fine-tuning completed\n";
	}


	void node_master_autoencoder::reconstruct()
	{
		//invio numero di esempi
		MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);

		//vettore delle unità visibili
		const uint n_visible_units = layers_size[0];

		auto input_units = my_vector<float>(n_visible_units);
		auto output_units = my_vector<float>(n_visible_units);

		//si prendono le immagini da IO
		smp_manager.path_folder = image_path_folder;
		smp_manager.restart();

		string file_name = "";

		//si ottengono displacements per gli accumulatori
		int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
		get_scatter_parts(send_counts, send_displacements, n_visible_units);

		//si ottengono displacements per gli accumulatori (questa volta le parti invisibili)
		int receive_counts[1 + total_accumulators], receive_displacements[1 + total_accumulators];
		get_scatter_parts(receive_counts, receive_displacements, n_visible_units);

		//errore medio quadratico
		float mean_root_squared_error = 0;

		for(uint i = 0; i != number_of_samples; i++)
		{
			smp_manager.get_next_sample(input_units,  default_extension.c_str() , &file_name);

			//invio esempio
			scatter_vector_sync(input_units, send_counts, send_displacements);

			//attesa ricezione risultato
			gather_vector_sync(output_units, receive_counts, receive_displacements);

			if(batch_mode)
			{
				//si salvano i risultati su file
				string new_image_path_folder = string(image_path_folder + "/output_rec" );
				smp_manager.save_sample(output_units, true, new_image_path_folder, file_name + default_extension);
			}
			else
			{
				//si mostra a video il risultato
				std::cout << "Showing original sample: '" <<  smp_manager.path_folder << "/" << file_name << "'\n";
				smp_manager.show_sample(input_units);

				std::cout << "Showing reconstructed sample:\n";
				smp_manager.show_sample(output_units);
			}

			mean_root_squared_error += root_squared_error(input_units, output_units);
		}

		std::cout << "Mean root squared error: " << mean_root_squared_error/number_of_samples << "\n";
	}


	string node_master_autoencoder::get_path_file(){

		return folder_parameters_path + "paral_master.txt";
	}


	void node_master_autoencoder::save_parameters()
	{
		string path_file = get_path_file();

		std::cout << "Saving autoencoder parameters to: '" + path_file + "'\n";

		// Create an input filestream
		std::ofstream myFile(path_file);

		// Make sure the file is open
		if(!myFile.is_open()) cout << "Could not open file: " + path_file << "\n";

		//salvataggio di numero rbm apprese e fine tuning effettuato
		myFile << "n_rbm," << trained_rbms << endl;
		myFile << "fine_tuning," << fine_tuning_finished << endl;

		myFile.close();
	}

	void node_master_autoencoder::load_parameters()
	{
		string path_file = get_path_file();

		std::cout << "Getting autoencoder parameters from: '" + path_file + "'\n";

		fine_tuning_finished = false;
		trained_rbms = 0;

		// Create an input filestream
		std::ifstream myFile(path_file);

		// Make sure the file is open
		if(!myFile.is_open()) cout << "Could not open file: " + path_file << "\n";

		// Helper vars
		std::string line;

		//Numero di RBM apprese
		if(std::getline(myFile, line))
		{
			// Create a stringstream of the current line
			std::stringstream ss(line);

			//riga dei bias nascosti
			ss.ignore(100, ',');
			ss >> trained_rbms;
		}

		//Fine tuning effettuato o meno
		if(std::getline(myFile, line))
		{
			// Create a stringstream of the current line
			std::stringstream ss(line);

			//riga dei bias nascosti
			ss.ignore(100, ',');
			ss >> fine_tuning_finished;
		}

		// Close file
		myFile.close();


		//Info generali
		std::cout << "Trained RBMs: " <<trained_rbms << "\n";
		std::cout << "Fine-tuning: " << (fine_tuning_finished ? "yes" : "no") << "\n";
	}


}
