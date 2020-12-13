#ifndef NODE_AUTOENCODER_H
#define NODE_AUTOENCODER_H


#include "custom_vectors.h"
#include "generators.h"
#include "MPI_adapters.h"


#include "samples_manager.h"


namespace parallel_autoencoder
{
	//Comandi implementati per l'autoencoder
	enum class CommandType {
		train = 0, exit = 1, load_pars = 2, save_pars = 3,
		reconstruct_image = 4 , delete_pars_file = 5, retry = 6
	};


    class node_autoencoder
    {

    protected:

    	const uint MAX_FOLDER_PARS_LENGTH = 300;

    	bool batch_mode;
    	bool reduce_io;
    	std::ostream& oslog;
		int mpi_rank;

    	//MPI pars
    	uint total_accumulators;
    	uint grid_rows;
    	uint grid_cols;

    	//each element indicates the grid orientation
    	my_vector<GridOrientation> orientation_grid;

    	//size of each layer
        my_vector<int> layers_size;

        uint number_of_rbm_to_learn;
        uint number_of_final_layers;

        uint number_of_samples = 0;

        //rbm pars
        uint trained_rbms;
        float rbm_momentum;
        uint rbm_n_training_epocs;
        uint rbm_size_minibatch;

        float rbm_initial_weights_variance;
		float rbm_initial_weights_mean;
		float rbm_initial_biases_value;

		//fine tuning pars
        bool fine_tuning_finished;
        float fine_tuning_learning_rate;
        float fine_tuning_n_training_epocs;

        string folder_parameters_path = "./autoencoder_pars/";



		//Questo metodo calcola il numero di elementi che (1) un accumulatore deve scambiare verso
		//i nodi della griglia oppure (2) gli elementi che un nodo della griglia deve scambiare
		//con ciascun accumulatore collegato.
		void calc_comm_sizes(const GridOrientation current_or, //orientamento comm
				my_vector<MPI_Comm_MasterSlave>& acc_gridcolrow_comm, //comm da accumulatore a righe o colonne
				const bool calc_for_acc, //indica se il calcolo va fatto per l'accumulatore o il nodo della griglia
				const uint my_k_col_row_number, //indice del nodo k o della cella in riga r o colonna c
				const int n_tot_vishid_units);


		//restituisce unità (visibili o hidden) per l'accumulatore k o il nodo della riga r o della colonna c
		static int get_units_for_node(const uint n_total_units, const uint total_nodes, const uint node_number);

		static float get_learning_rate_rbm(const uint epoch_number, const uint layer_number);

    public:

        node_autoencoder(const my_vector<int>& _layers_size,
        		uint _total_accumulators, uint _grid_row, uint _grid_col,
				uint rbm_n_epochs, uint finetuning_n_epochs, uint rbm_batch_size, bool batch_mode, bool _reduce_io,
				std::ostream& _oslog, int _mpi_rank);

        virtual ~node_autoencoder();


        void loop();
        void execute_command(CommandType command);
        virtual CommandType wait_for_command();


        virtual void train_rbm() = 0;
        virtual void fine_tuning() = 0;

        virtual void reconstruct() = 0;

        virtual string get_path_file() = 0;
        virtual void save_parameters() = 0;
        virtual void load_parameters() = 0;



    private:

        void set_size_for_layers(const my_vector<int>& _layers_size_source);
		void set_orientation_for_layers();
    };
}





#endif /* NODE_AUTOENCODER_H */
