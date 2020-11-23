#ifndef NODE_AUTOENCODER_H
#define NODE_AUTOENCODER_H


#include <vector>
#include <random>
#include <string>
#include "mpi.h"

using std::vector;


namespace parallel_autoencoder{

//todo
//migliorare efficienza utilizzando matrici invece che vettori
//fare un check sui cicli for utilizzando i tipi di interi corretti e l'operatore != invece che <
//gestire possibilità di ricevere altro dai buffer mentre si calcolano i dati
    class node_autoencoder
    {
    	//enum NodeType { master, accumulator, cell};

    protected:

    	MPI_Datatype mpi_datatype_tosend = MPI_FLOAT;

    	std::ostream& oslog;
		int mpi_rank;

    	enum GridOrientation { row_first, col_first };
    	enum CommandType { train, exit };

    	//MPI pars
    	int total_accumulators;
    	int grid_rows;
    	int grid_cols;


    	//each element indicates the grid orientation
    	vector<GridOrientation> orientation_grid;

        std::default_random_engine generator;

    	//size of each layer
        vector<int> layers_size;

        int number_of_rbm_to_learn;
        int number_of_final_layers;

        int number_of_samples = 0;

        //rbm pars
        int trained_rbms;
        float rbm_learning_rate;
        float rbm_momentum;
        int rbm_n_training_epocs;
        int rbm_size_minibatch;

        float rbm_initial_weights_variance;
		float rbm_initial_weights_mean;
		float rbm_initial_biases_value;

		//fine tuning pars
        bool fine_tuning_finished;
        float fine_tuning_learning_rate;
        float fine_tuning_n_training_epocs;


    public:

        node_autoencoder(const vector<int>& _layers_size, std::default_random_engine& _generator,
        		int _total_accumulators, int _grid_row, int _grid_col,
				std::ostream& _oslog, int _mpi_rank)
        :oslog{_oslog}
        {
        	mpi_rank = _mpi_rank;

        	total_accumulators =_total_accumulators;
			grid_rows =_grid_row;
			grid_cols = _grid_col;

			generator = _generator;

			number_of_rbm_to_learn = _layers_size.size() - 1;
			number_of_final_layers = _layers_size.size() * 2 - 1;

			//_layers_size contiene la grandezza dei layer fino a quello centrale
			//in fase di rollup verranno creati altri layer per la ricostruzione
			layers_size = vector<int>(number_of_final_layers);
			for(uint i = 0; i !=_layers_size.size(); i++)
			{
			   layers_size[i] = _layers_size[i];

			   //si copia la grandezza del layer per il layer da ricostruire
			   int rec_layer = number_of_final_layers - i - 1;
			   layers_size.at(rec_layer) = layers_size[i];
			}

			//si determina l'orientamento della griglia per ciascun layer
			//di default le righe prendono gli input e le colonne gli output
			orientation_grid = vector<GridOrientation>(number_of_final_layers - 1);
			for(int i = 0; i != number_of_rbm_to_learn; i++)
			{
				int input_nodes = layers_size[i];
				int output_nodes = layers_size[i + 1];

				//c'è da considerare il fatto che l'orientamento è deciso in base al costo della comunicazione
				//di fatto significa se se i nodi di output sono maggiori di quelli di input, allora si inverte l'orientamento
				GridOrientation orientation = 3 * input_nodes < 4 * output_nodes ? col_first : row_first;

				orientation_grid[i] = orientation;

				//dato che il rollup si basa sul riutilizzo dei pesi, e dato che il
				//numero di nodi di input e di output è invertito tra il layer di ricostruzione e quello di encoding
				//l'orientamento sarà necessariamente inverso
				int rec_layer = number_of_final_layers - i - 2;
				orientation_grid[rec_layer] = orientation == col_first ? row_first : col_first;
			}

			trained_rbms = 0;
			rbm_learning_rate = 0.01; //todo al primo layer è diverso
			rbm_momentum = 0.9;
			rbm_n_training_epocs = 20;//todo sistemare
			rbm_size_minibatch = 20;//todo sistemare

			rbm_initial_weights_variance = 0.01;
			rbm_initial_weights_mean = 0;
			rbm_initial_biases_value = 0;


			fine_tuning_n_training_epocs = 5;
			fine_tuning_learning_rate = 10e-6;

			fine_tuning_finished = false;
	   }


        void loop()
        {
        	CommandType command;
        	do
        	{
        		//ottengo il comando (dipende dalla classe)
        		command = wait_for_command();
        		oslog << "My command is " << command << "\n";

        		switch(command){

					case train:

						//1. Si apprendono le RBM per ciascun layer
						oslog << "Imparando le RBM...\n";
						oslog << "Numero di RBM da apprendere: " <<  number_of_rbm_to_learn <<"\n";
						oslog << "Numero di RBM gia apprese: " << trained_rbms << "\n";
						oslog << "Numero di layer finali: " <<  number_of_final_layers <<"\n";

						train_rbm();

						if(!fine_tuning_finished)
							fine_tuning();

					break;

					case exit:
						break;
        		}
        	}
        	while(command != exit);
        }

        //restituisce unità (visibili o hidden) per l'accumulatore k o il nodo della riga r o della colonna c
        static int get_units_for_node(const uint n_total_units, const uint total_nodes, const uint node_number)
        {
        	const int n_units_x_node = ceil((float)n_total_units / total_nodes);

        	const int overflow_units = (node_number + 1) * n_units_x_node - n_total_units;

        	int n_my_units = n_units_x_node;
			if(overflow_units > 0)
			{
				n_my_units -= overflow_units;
				if(n_my_units < 0) n_my_units = 0; //caso limite
			}

			return n_my_units;
        }

        //Questo metodo calcola il numero di elementi che (1) un accumulatore deve scambiare verso
		//i nodi della griglia oppure (2) gli elementi che un nodo della griglia deve scambiare
		//con ciascun accumulatore collegato.
		void calc_comm_sizes(const GridOrientation current_or, //orientamento comm
				vector<MP_Comm_MasterSlave>& acc_gridcolrow_comm, //comm da accumulatore a righe o colonne
				const bool calc_for_acc, //indica se il calcolo va fatto per l'accumulatore o il nodo della griglia
				const uint my_k_col_row_number, //indice del nodo k o della cella in riga r o colonna c
				const int n_tot_vishid_units)
		{
			auto total_grid_rowcol = (current_or == row_first ? grid_rows : grid_cols);

			uint index_acc = 0;
			uint index_gridnode = 0;

			uint elements_for_acc = get_units_for_node(n_tot_vishid_units, total_accumulators, index_acc);
			uint elements_for_grid_node = get_units_for_node(n_tot_vishid_units, total_grid_rowcol, index_gridnode);

			do
			{
				if(calc_for_acc)
				{
					//se questo calcolo lo si sta facendo per l'accumulatore...
					if(index_acc == my_k_col_row_number)
					{
						//ora abbiamo ottenuto il numero di elementi che l'accumulatore deve dare verso una determinata riga/colonna
						//si cerca il comunicatore della riga/colonna di riferimento
						for(auto& comm : acc_gridcolrow_comm)
							if(comm.row_col_id == index_gridnode)
							{
								comm.n_items_to_send = std::min(elements_for_acc, elements_for_grid_node);

								//log
								oslog << "I'm a k node, I will send " << comm.n_items_to_send <<
										" elements to each node of the " << current_or << " nodes\n";
								break;
							}
					}
				}
				else
				{
					//se questo calcolo lo si sta facendo per il nodo della cella ...
					if(index_gridnode == my_k_col_row_number)
					{
						//ora abbiamo ottenuto il numero di elementi che l'accumulatore deve dare verso una determinata riga/colonna
						//si cerca il comunicatore della riga/colonna di riferimento
						for(auto& comm : acc_gridcolrow_comm)
							if(comm.root_id == index_acc)
							{
								comm.n_items_to_send = std::min(elements_for_acc, elements_for_grid_node);

								//log
								oslog << "I'm a cell node, I will send " << comm.n_items_to_send <<
										" elements to the k node (" << current_or  << ")\n";
								break;
							}
					}
				}

				auto diff = elements_for_acc - elements_for_grid_node;

				if(diff >= 0)
				{
					if(diff > 0)
						elements_for_acc = diff;

					index_gridnode++;
					elements_for_grid_node = get_units_for_node(n_tot_vishid_units, total_grid_rowcol, index_gridnode);
				}

				if(diff <= 0)
				{
					if(diff < 0)
						elements_for_grid_node = diff * -1;

					index_acc++;
					elements_for_acc = get_units_for_node(n_tot_vishid_units, total_accumulators, index_acc);
				}

			}
			while((calc_for_acc ? index_acc : index_gridnode) <= my_k_col_row_number);
			//non mi fermo fintantoché non ho analizzato tutti i possibili collegamenti del nodo
		}



        virtual CommandType wait_for_command() = 0;

        virtual void train_rbm() = 0;
        virtual void fine_tuning() = 0;




        //virtual vector<bool> encode() = 0;
        virtual vector<float> reconstruct() = 0;

        virtual void save_parameters() = 0;
        virtual void load_parameters() = 0;

        //virtual ~node_autoencoder()=0;
    };
}





#endif /* NODE_AUTOENCODER_H */
