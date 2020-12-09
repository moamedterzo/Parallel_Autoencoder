/*
 * node_cell_autoencoder.h
 *
 *  Created on: 19 nov 2020
 *      Author: giovanni
 */

#ifndef NODE_CELL_AUTOENCODER_H_
#define NODE_CELL_AUTOENCODER_H_



#include "node_autoencoder.h"


namespace parallel_autoencoder
{

	class node_cell_autoencoder : public node_autoencoder{

	public:

		node_cell_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
				uint _total_accumulators, uint _grid_row, uint _grid_col,
				uint rbm_n_epochs, uint finetuning_n_epochs, uint rbm_batch_size, bool batch_mode, bool _reduce_io,
				std::ostream& _oslog, int _mpi_rank,
				uint _row_number, uint _col_number,
				my_vector<MPI_Comm_MasterSlave>& _accs_row_comm, my_vector<MPI_Comm_MasterSlave>& _accs_col_comm);


		void train_rbm();

	    void save_new_samples(
	    		MPI_Req_Manager_Cell& reqVis, MPI_Req_Manager_Cell& reqHid,
				matrix<float>& weights,
	    		my_vector<float>& visible_units1, my_vector<float>& visible_units2,
				my_vector<float>& hidden_units1, my_vector<float>& hidden_units2);


		void fine_tuning();
	    void reconstruct();

	    string get_path_file();

		void save_parameters();
	    void load_parameters();

	private:

			my_vector<matrix<float>> layers_weights;

			uint row_number;
			uint col_number;

			//comunicators for grid-accumulators
			my_vector<MPI_Comm_MasterSlave> accs_row_comm;
			my_vector<MPI_Comm_MasterSlave> accs_col_comm;

			//communicators for each layer
			my_vector<my_vector<MPI_Comm_MasterSlave>> acc_hid_comm_for_layer;
			my_vector<my_vector<MPI_Comm_MasterSlave>> acc_vis_comm_for_layer;



			void calc_all_comm_sizes();
			void get_my_visible_hidden_units(const uint layer_number, uint& n_my_visible_units, uint& n_my_hidden_units);

			void rollup_for_weights();

			void get_activation_output_layers(my_vector<my_vector<float>>& activation_layers, my_vector<my_vector<float>>& output_layers);


		    void forward_pass(my_vector<my_vector<float>>& activation_layers, my_vector<my_vector<float>>& output_layers);
		    void backward_pass(my_vector<my_vector<float>>& activation_layers, my_vector<my_vector<float>>& output_layers);

	};

}


#endif /* NODE_CELL_AUTOENCODER_H_ */
