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

	class node_master_autoencoder : public node_autoencoder
	{

	public:
		node_master_autoencoder(const my_vector<int>& _layers_size,
					uint _total_accumulators, uint _grid_row, uint _grid_col,
					uint rbm_n_epochs, uint finetuning_n_epochs, uint rbm_batch_size, bool batch_mode, bool _reduce_io,
					std::ostream& _oslog, int _mpi_rank,
					MPI_Comm& _master_accs_comm,
					samples_manager& _smp_manager);


		CommandType wait_for_command();



		void train_rbm();


		void save_new_samples(const uint layer_number,const uint n_visible_units,const uint n_hidden_units,
				const char *sample_extension,
				my_vector<float>& visible_units, my_vector<float>& visible_units_send_buffer);


		void fine_tuning();
		void reconstruct();

		string get_path_file();

		void save_parameters();
		void load_parameters();


	private:

		//comunicatore da master a nodi accumulatori
		MPI_Comm master_accs_comm;

		//gestore degli esempi input/output su disco
		samples_manager smp_manager;

		//percorso della cartella che contiene le immagini iniziali
		string image_path_folder;



		void scatter_vector(const my_vector<float>& vec, const int send_counts[], const int displs[], MPI_Request *reqSend);
		void scatter_vector_sync(const my_vector<float>& vec, const int send_counts[], const int displs[]);

		void gather_vector(const my_vector<float>& vec, const int receive_counts[], const int displs[], MPI_Request *reqRecv);
		void gather_vector_sync(const my_vector<float>& vec, const int receive_counts[], const int displs[]);

		void get_scatter_parts(int counts[], int displacements[], const int n_total_units);
	};
}


#endif /* NODE_MASTER_AUTOENCODER_H_ */
