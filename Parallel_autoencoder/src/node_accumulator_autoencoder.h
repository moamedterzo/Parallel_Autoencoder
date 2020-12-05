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



namespace parallel_autoencoder
{

	class node_accumulator_autoencoder : public node_autoencoder
	{

	public:

			node_accumulator_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
					uint _total_accumulators, uint _grid_row, uint _grid_col,
					uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode, bool _reduce_io,
					std::ostream& _oslog, uint _mpi_rank,
					uint _k_number,
					MPI_Comm& _master_accs_comm,
					my_vector<MPI_Comm_MasterSlave>& _acc_rows_comm, my_vector<MPI_Comm_MasterSlave>& _acc_cols_comm);

			void train_rbm();

			void save_new_samples(
					 MPI_Req_Manager_Accumulator& reqVis, MPI_Req_Manager_Accumulator& reqHid,
					 MPI_Request *reqMaster, my_vector<float>& hidden_biases,
					 my_vector<float>& visible_units1, my_vector<float>& visible_units2,
					 my_vector<float>& hidden_units1, my_vector<float>& hidden_units2);



			void fine_tuning();
			void reconstruct();

		    string get_path_file();

			void save_parameters();
			void load_parameters();


	private:
		my_vector<my_vector<float>> layer_biases;

		uint k_number;

		//master-accumulators communicator
		MPI_Comm master_accs_comm;

		//comunicators for grid-accumulators
		my_vector<MPI_Comm_MasterSlave> acc_rows_comm;
		my_vector<MPI_Comm_MasterSlave> acc_cols_comm;

		//communicators for each layer
		my_vector<my_vector<MPI_Comm_MasterSlave>> acc_hid_comm_for_layer;
		my_vector<my_vector<MPI_Comm_MasterSlave>> acc_vis_comm_for_layer;


		void calc_all_comm_sizes();
		void get_my_visible_hidden_units(const uint layer_number, uint& n_my_visible_units, uint& n_my_hidden_units);

		my_vector<my_vector<float>> get_activation_layers();

		void forward_pass(my_vector<my_vector<float>>& activation_layers);
		void backward_pass(my_vector<my_vector<float>>& activation_layers);


		void receive_from_master(my_vector<float>& vis_vec, MPI_Request *reqVis);
		void receive_from_master_sync(my_vector<float>& vis_vec);
		void send_to_master(my_vector<float>& hid_vec, MPI_Request *reqHid);
		void send_to_master_sync(my_vector<float>& hid_vec);


	};

}
#endif /* NODE_ACCUMULATOR_AUTOENCODER_H_ */
