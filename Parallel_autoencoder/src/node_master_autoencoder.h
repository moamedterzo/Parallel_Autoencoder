/*
 * node_master_autoencoder.h
 *
 *  Created on: 19 nov 2020
 *      Author: giovanni
 */

#ifndef NODE_MASTER_AUTOENCODER_H_
#define NODE_MASTER_AUTOENCODER_H_



#include "node_autoencoder.h"


namespace parallel_autoencoder{

	class node_master_autoencoder : public node_autoencoder{

	private:

		//comunicatore da master a nodi accumulatori
		MP_Comm_MasterSlave master_accs_comm;

		//gestore degli esempi input/output su disco
		samples_manager smp_manager;

		//percorso della cartella che contiene le immagini iniziali
		string image_path_folder;



		void ScatterInputVector(const my_vector<float>& vec, const int send_counts[], const int displs[], MPI_Request *reqSend)
		{
			MPI_Iscatterv(vec.data(), send_counts, displs, mpi_datatype_tosend,
						nullptr, 0,  mpi_datatype_tosend,
						0, master_accs_comm.comm, reqSend);
		}

		void ReceiveOutputVector(const my_vector<float>& vec, const int receive_counts[], const int displs[], MPI_Request *reqRecv)
		{
			MPI_Igatherv(MPI_IN_PLACE, 0, mpi_datatype_tosend,
					vec.data(), receive_counts, displs, mpi_datatype_tosend,
					0, master_accs_comm.comm,  reqRecv);
		}

		void GetScatterParts(int counts[], int displacements[], const int n_total_units)
		{
			//root non riceve nulla
			counts[0] = displacements[0] = 0;

			uint n_units_for_acc = 0;
			for(uint k = 0; k != total_accumulators; k++){

				//l'accumulatore k è posizionato alla posizione k + 1 (a causa dell'indice 0 che rappresenta il root)
				//posizionamento iniziale
				displacements[k + 1] = displacements[k] + n_units_for_acc;

				//calcolo elementi da inviare/ricevere per l'accumulatore
				n_units_for_acc = get_units_for_node(n_total_units, total_accumulators, k);
				counts[k + 1] = n_units_for_acc;
			}
		}

	public:
		node_master_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
					uint _total_accumulators, uint _grid_row, uint _grid_col,
					std::ostream& _oslog, int _mpi_rank,
					MP_Comm_MasterSlave& _master_accs_comm,
					samples_manager& _smp_manager)
			: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
			{
				smp_manager = _smp_manager;
				master_accs_comm = _master_accs_comm;

				image_path_folder = string(smp_manager.path_folder);
			}


		CommandType wait_for_command()
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

				//indico il numero di elementi del dataset
				number_of_samples = smp_manager.get_number_samples();

				//Todo Al momento il numero di esempi deve essere necessariamente pari
				if(number_of_samples % 2 != 0)
				{
					string message = "Number of samples must be even";
					std::cout << message << "\n";

					throw new std::runtime_error(message);
				}

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
			if(command == CommandType::train)
			{
				//invio numero di esempi
				MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);
			}
			else if(command == CommandType::load_pars || command == CommandType::save_pars)
			{
				//invio cartella che contiene i parametri
				if(strcmp(char_path_file, ".") != 0)
					folder_parameters_path = string(char_path_file);

				MPI_Bcast(char_path_file, MAX_FOLDER_PARS_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
			}


			return command;
		}





		void train_rbm()
		{
			//Per ciascun layer...
			//se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
			for(uint layer_number = trained_rbms; layer_number != number_of_rbm_to_learn; layer_number++)
			{
				const uint n_visible_units = layers_size[layer_number];
				const uint n_hidden_units = layers_size[layer_number + 1];
				const char *sample_extension = layer_number == 0 ? ".jpg" : ".txt";

				//si ottengono displacements per gli accumulatori
				int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
				GetScatterParts(send_counts, send_displacements, n_visible_units);


				std::cout << "-- Imparando il layer numero: " << layer_number
						<< ", hidden units: " << n_hidden_units
						<< ", visible units: " << n_visible_units << " --\n";


				MPI_Request reqSend;
				my_vector<float> visible_units(n_visible_units);
				my_vector<float> visible_units_send_buffer(n_visible_units);

				//si avvia il processo di apprendimento per diverse epoche
				for(uint epoch = 0; epoch != rbm_n_training_epocs; epoch++){

					if(epoch % 5 == 0)
						std::cout << "Training epoch: " << epoch << "\n";

					//per ciascun esempio leggo da file system mentre invio l'esempio precedente
					for(uint current_sample = 0; current_sample != number_of_samples; current_sample++)
					{
						//lettura file system
						smp_manager.get_next_sample(visible_units, sample_extension);

						//attendo completamento dell'invio precedente per poter inviare il prossimo vettore
						if(current_sample != 0)
							MPI_Wait(&reqSend, MPI_STATUS_IGNORE);

						visible_units_send_buffer = visible_units;

						ScatterInputVector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);
					}

					//si riavvia l'ottenimento dei samples
					smp_manager.restart();

					//si conclude l'ultimo invio effettuato
					MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
				}


				std::cout<< "Fine apprendimento singola RBM\n";

				//SALVATAGGIO NUOVI INPUT
				save_new_samples(layer_number, n_visible_units, n_hidden_units, sample_extension,
						visible_units, visible_units_send_buffer);

				//contatore che memorizza il numero di rbm apprese
				trained_rbms++;
				save_parameters();
			}

		}


		inline void save_new_samples(const uint layer_number,const uint n_visible_units,const uint n_hidden_units,
				const char *sample_extension,
				my_vector<float>& visible_units, my_vector<float>& visible_units_send_buffer)
		{
			//si deve salvare sul disco i risultati di attivazione del layer successivo
			//essi saranno utilizzati come input per la prossima fare di training
			string new_image_path_folder = string(image_path_folder + "/" + std::to_string(layer_number));
			std::cout << "Salvando i risultati intermedi per il prossimo step nella cartella '"	<< new_image_path_folder << "'\n";

			//nome del file dove salvare ciascun esempio
			string sample_filename, sample_filename_prec;

			//risultato dell'operazione
			my_vector<float> output_samples(n_hidden_units, 0.0);

			//si ottengono displacements per gli accumulatori (questa volta le parti invisibili)
			int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
			GetScatterParts(send_counts, send_displacements, n_visible_units);

			int receive_counts[1 + total_accumulators], receive_displacements[1 + total_accumulators];
			GetScatterParts(receive_counts, receive_displacements, n_hidden_units);

			//si fa in modo di ottimizzare il processo di scambio dei dati
			MPI_Request reqSend, reqRecv;

			//Si inizia a rileggere ciascun input e inviarlo
			smp_manager.restart();


			for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
			{
				//lettura esempio da file
				smp_manager.get_next_sample(visible_units, sample_extension, &sample_filename);

				if(current_index_sample > 0)
				{
					//attendo invio esempio precedente
					MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
				}

				//invio esempio (copia valori nel buffer)
				visible_units_send_buffer = visible_units;
				ScatterInputVector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);


				if(current_index_sample > 0)
				{
					//attesa gather
					ReceiveOutputVector(output_samples, receive_counts, receive_displacements,&reqRecv);
					MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

					//si salva su file il vettore Hidden ottenuto
					smp_manager.save_sample(output_samples, false, new_image_path_folder, sample_filename_prec + ".txt"); //dati in formato testuale
					smp_manager.save_sample(output_samples, true, new_image_path_folder, sample_filename_prec+ ".jpg"); //dati in formato immagine
				}

				//dato che vengono letti prima due input e poi si inizia a salvare, è necessario memorizzare il nome del file precedente
				sample_filename_prec = sample_filename;
			}

			//migliorare codice
			//attesa gather
			ReceiveOutputVector(output_samples, receive_counts, receive_displacements,&reqRecv);
			MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

			//si salva su file il vettore Hidden ottenuto
			smp_manager.save_sample(output_samples, false, new_image_path_folder, sample_filename_prec + ".txt"); //dati in formato testuale
			smp_manager.save_sample(output_samples, true, new_image_path_folder, sample_filename_prec+ ".jpg"); //dati in formato immagine


			//in maniera del tutto trasparente si utilizzerà questo nuovo percorso per ottenere i dati in input
			smp_manager.path_folder = new_image_path_folder;
			smp_manager.restart();
		}




		void fine_tuning()
		{
			std::cout << "Inizio fine-tuning\n" ;

			 //si passa nuovamente alle immagini iniziali
			smp_manager.path_folder = image_path_folder;

			//vettore delle unità visibili
			const uint n_visible_units = layers_size[0];

			auto visible_units = my_vector<float>(layers_size[0]);
			auto visible_units_send_buffer = my_vector<float>(layers_size[0]);
			MPI_Request reqSend;

			//si ottengono displacements per gli accumulatori
			int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
			GetScatterParts(send_counts, send_displacements, n_visible_units);


			//per ogni epoca...
			for(uint epoch = 0; epoch != fine_tuning_n_training_epocs; epoch++)
			{
				smp_manager.restart();
				std::cout << "Training epoch: " << epoch << "\n";

				for(uint current_index_sample = 0; current_index_sample != number_of_samples; current_index_sample++)
				{
					//lettura esempio da file
					smp_manager.get_next_sample(visible_units, ".jpg");

					if(current_index_sample != 0)
					{
						//attendo invio esempio precedente
						MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
					}

					//invio esempio (copia valori nel buffer)
					visible_units_send_buffer = visible_units;
					ScatterInputVector(visible_units_send_buffer, send_counts, send_displacements, &reqSend);
				}

				//attendo invio esempio precedente
				MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
			} //fine epoca


			//allenamento concluso
			fine_tuning_finished = true;
			save_parameters();

			//si aspettano eventuali nodi rimasti indietro
			MPI_Barrier(MPI_COMM_WORLD);

			std::cout << "Fine-tuning completed\n";
		}


		my_vector<float> reconstruct()
		{

			//vettore delle unità visibili
			const uint n_visible_units = layers_size[0];

			auto input_units = my_vector<float>(n_visible_units);
			auto output_units = my_vector<float>(n_visible_units);

			//si prende l'immagine dal manager
			smp_manager.path_folder = image_path_folder;
			smp_manager.restart();

			string file_name = "";
			smp_manager.get_next_sample(input_units, ".jpg", &file_name);

			//si ottengono displacements per gli accumulatori
			int send_counts[1 + total_accumulators], send_displacements[1 + total_accumulators];
			GetScatterParts(send_counts, send_displacements, n_visible_units);

			//invio esempio
			MPI_Request reqSend;
			ScatterInputVector(input_units, send_counts, send_displacements, &reqSend);


			//si ottengono displacements per gli accumulatori (questa volta le parti invisibili)
			int receive_counts[1 + total_accumulators], receive_displacements[1 + total_accumulators];
			GetScatterParts(receive_counts, receive_displacements, n_visible_units);

			//attesa ricezione risultato
			MPI_Request reqRecv;
			ReceiveOutputVector(output_units, receive_counts, receive_displacements, &reqRecv);
			MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

			//si mostra a video il risultato
			std::cout << "Showing original sample: '" <<  smp_manager.path_folder << "/" << file_name << "'\n";
			smp_manager.show_sample(input_units);

			std::cout << "Showing reconstructed sample:\n";
			smp_manager.show_sample(output_units);


			return output_units;
		}


		string get_path_file(){

			return folder_parameters_path + "paral_master.txt";
		}


		void save_parameters()
		{
			string path_file = get_path_file();

			std::cout << "Saving autoencoder parameters to: '" + path_file + "'\n";

			// Create an input filestream
			std::ofstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

			//salvataggio di numero rbm apprese e fine tuning effettuato
			myFile << "n_rbm," << trained_rbms << endl;
			myFile << "fine_tuning," << fine_tuning_finished << endl;

			myFile.close();
		}

		void load_parameters()
		{
			string path_file = get_path_file();

			std::cout << "Getting autoencoder parameters from: '" + path_file + "'\n";

			fine_tuning_finished = false;
			trained_rbms = 0;

			// Create an input filestream
			std::ifstream myFile(path_file);

			// Make sure the file is open
			if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

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
	};
}


#endif /* NODE_MASTER_AUTOENCODER_H_ */
