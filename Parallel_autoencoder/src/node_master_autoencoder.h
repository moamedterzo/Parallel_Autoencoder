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
		samples_manager& smp_manager;

	public:
		node_master_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
					int _total_accumulators, int _grid_row, int _grid_col,
					std::ostream& _oslog, int _mpi_rank,
					MP_Comm_MasterSlave _master_accs_comm,
					samples_manager& _smp_manager)
			: node_autoencoder(_layers_size, _generator, _total_accumulators, _grid_row, _grid_col, _oslog, _mpi_rank)
		    ,smp_manager{_smp_manager}
			{
				master_accs_comm = _master_accs_comm;
				smp_manager = _smp_manager;
			}

		CommandType wait_for_command()
		{
			CommandType command;

			std::cout << "Hello, I'm the master, what do you want to do?\n";
			std::cout << "1 to train a rbm\n";
			std::cout << "-1 to exit\n";

			int res;
			std::cin >> res;

			if(res == 1)
			{
				command = train;
			}

			else if(res == -1)
			{
				command = exit;
			}

			//alert other node
			MPI_Bcast(&command,1, MPI_INT, 0, MPI_COMM_WORLD);

			//indico il numero di elementi del dataset
			number_of_samples = smp_manager.get_number_samples();

			//Todo Al momento il numero di esempi deve essere necessariamente pari
			if(number_of_samples % 2 != 0)
				throw new std::exception();

			MPI_Bcast(&number_of_samples,1, MPI_INT, 0, MPI_COMM_WORLD);

			return command;
		}


		void ScatterInputVector(vector<float>& vec, int n_units_x_accumulator, MPI_Request *reqSend)
		{
			MPI_Iscatter(vec.data(), n_units_x_accumulator, mpi_datatype_tosend,
							MPI_IN_PLACE, 0, NULL,
							master_accs_comm.root_id, master_accs_comm.comm, reqSend);
		}


		void train_rbm()
		{
			//percorso della cartella che contiene le immagini iniziali
			string image_path_folder = string(smp_manager.path_folder);

			//Per ciascun layer...
			//se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
			for(int layer_number = trained_rbms; layer_number < number_of_rbm_to_learn; layer_number++)
			{
				const int n_visible_units = layers_size[layer_number];
				const int n_hidden_units = layers_size[layer_number + 1];

				//numero massimo di unità visibili per singolo acc (l'ultimo acc potrebbe averne di meno)
				const int n_visible_units_x_accumulator = ceil((float)n_visible_units / total_accumulators);

				std::cout << "-- Imparando il layer numero: " << layer_number
						<< ", hidden units: " << n_hidden_units
						<< ", visible units: " << n_visible_units << " --\n";


				MPI_Request reqSend;
				vector<float> visible_units(n_visible_units);
				vector<float> visible_units_send_buffer(n_visible_units);

				//si avvia il processo di apprendimento per diverse epoche
				for(int epoch = 0; epoch < rbm_n_training_epocs; epoch++){

					if(epoch % 1 == 0)
						std::cout << "Training epoch: " << epoch << "\n";

					//per ciascun esempio leggo da file system mentre invio l'esempio precedente
					for(int current_sample = 0; current_sample < number_of_samples; current_sample++)
					{
						if(current_sample % 100 == 0)
							std::cout << "current_index_sample: " << current_sample << "\n";

						//lettura file system
						smp_manager.get_next_sample(visible_units);

						//attendo completamento dell'invio precedente per poter inviare il prossimo vettore
						if(current_sample != 0)
							MPI_Wait(&reqSend, nullptr);

						visible_units_send_buffer = visible_units;
						ScatterInputVector(visible_units_send_buffer, n_visible_units_x_accumulator, &reqSend);
					}

					//si riavvia l'ottenimento dei samples
					smp_manager.restart();

					//si conclude l'ultimo invio effettuato
					MPI_Wait(&reqSend, nullptr);
				}


				std::cout<< "Fine apprendimento RBM\n";

				//SALVATAGGIO NUOVI INPUT
				{
					//si deve salvare sul disco i risultati di attivazione del layer successivo
					//essi saranno utilizzati come input per la prossima fare di training
					string new_image_path_folder = string(image_path_folder + "/" + std::to_string(layer_number));
					std::cout << "Salvando i risultati intermedi per il prossimo step nella cartella '"	<< new_image_path_folder << "'\n";

					//nome del file dove salvare ciascun esempio
					string sample_filename;

					//risultato dell'operazione
					vector<float> output_samples(n_hidden_units);

					//si fa in modo di ottimizzare il processo di scambio dei dati
					MPI_Request reqSend, reqRecv;

					//Si inizia a rileggere ciascun input e inviarlo
					smp_manager.restart();

					bool continue_read = smp_manager.get_next_sample(visible_units, &sample_filename);
					if(continue_read)
					{
						visible_units_send_buffer = visible_units;
						ScatterInputVector(visible_units_send_buffer, n_visible_units_x_accumulator, &reqSend);
					}

					while(continue_read)
					{
						MPI_Igather(MPI_IN_PLACE, 0, NULL,
								output_samples.data(), output_samples.size(), mpi_datatype_tosend,
								master_accs_comm.root_id, master_accs_comm.comm, &reqRecv);

						//lettura prossimo esempio e invio non appena è completato l'invio precedente
						continue_read = smp_manager.get_next_sample(visible_units, &sample_filename);
						MPI_Wait(&reqSend, nullptr);

						if(continue_read)
						{
							visible_units_send_buffer = visible_units;
							ScatterInputVector(visible_units_send_buffer, n_visible_units_x_accumulator, &reqSend);
						}

						 MPI_Wait(&reqRecv, nullptr);

						 //si salva su file il vettore Hidden ottenuto
						 smp_manager.save_sample(output_samples, new_image_path_folder, sample_filename);
					}

					//in maniera del tutto trasparente si utilizzerà questo nuovo percorso per ottenere i dati in input
					smp_manager.path_folder = new_image_path_folder;
					smp_manager.restart();

					//contatore che memorizza il numero di rbm apprese
					trained_rbms++;
					save_parameters();
				}
			}

		}

		void fine_tuning()
		{
			std::cout << "Inizio fine-tuning\n" ;

		}

		vector<float> reconstruct(){}

		void save_parameters()
		{
			//nessun parametro da salvare
		}

		void load_parameters()
		{
			//nessun parametro da caricare
		}
	};
}


#endif /* NODE_MASTER_AUTOENCODER_H_ */
