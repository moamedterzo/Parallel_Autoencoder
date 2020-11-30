/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   main.cpp
 * Author: Amedeo
 *
 * Created on 14 novembre 2020, 09:33
 */

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <random>
#include <string>


#include "custom_vectors.h"
#include "samples_manager.h"
#include "node_accumulator_autoencoder.h"
#include "node_master_autoencoder.h"
#include "node_cell_autoencoder.h"
#include "node_single_autoencoder.h"

using namespace std;
using namespace parallel_autoencoder;



//numero di elementi per layer predeterminato
my_vector<int> layers_size { 4096, 16, 128, 64, 32};

uint k_accumulators = 4, grid_total_rows = 4, grid_total_cols = 2;

string path_dataset = "./mnist_chinese/data";
uint number_of_samples = 2;
uint rbm_n_epochs = 80;
uint finetuning_n_epochs = 5;
bool batch_mode = false;




//rank MPI id
int myid;




void get_single_arg(std::string& s, const std::string& arg_name, uint& var_to_set)
{
    std::size_t pos = s.find(arg_name);

    if (pos != std::string::npos)
    {
    	var_to_set = std::stoi(s.replace(pos, arg_name.length(), ""));
    }
}

void get_single_arg(std::string& s, const std::string& arg_name, string& var_to_set)
{
    std::size_t pos = s.find(arg_name);

    if (pos != std::string::npos)
    {
    	var_to_set = s.replace(pos, arg_name.length(), "");
    }
}

void get_single_arg(std::string& s, const std::string& arg_name, bool& var_to_set)
{
    std::size_t pos = s.find(arg_name);

    if (pos != std::string::npos)
    {
    	var_to_set = std::stoi(s.replace(pos, arg_name.length(), "")) == 1;
    }
}

void get_layer_sizes(std::string& s, const std::string& arg_name)
{
    std::size_t pos = s.find(arg_name);

    if (pos != std::string::npos)
    {
    	//valori splittati da una virgola
    	auto values = s.replace(pos, arg_name.length(), "");
    	static const string delimiter = ",";

    	//ottenendo i valori
    	layers_size = my_vector<int>(0);

    	pos = 0;
    	std::string token;
    	while ((pos = s.find(delimiter)) != std::string::npos) {
    	    token = s.substr(0, pos);
    	    s.erase(0, pos + delimiter.length());

    	    layers_size.push_back(std::stoi(token));
    	}

    	layers_size.push_back(std::stoi(s));
    }
}

void parse_args(const int argc, char** argv)
{
	static const string arg_k_accumulators = "--k_acc:";
	static const string arg_g_rows = "--g_rows:";
	static 	const string arg_g_cols = "--g_cols:";
	static const string arg_rbm_n_epochs = "--rbm_epochs:";
	static const string arg_finetuning_n_epochs = "--fn_epochs:";
	static const string arg_batch_mode = "--batch:";
	static const string arg_n_samples = "--n_samples:";
	static const string arg_path_dataset = "--path_dataset:";
	static const string arg_layers_sizes = "--layer_sizes:";

	for(int i = 0; i < argc; i++)
	{
		string val_arg = string(argv[i]);

		get_single_arg(val_arg, arg_k_accumulators, k_accumulators);
		get_single_arg(val_arg, arg_g_rows, grid_total_rows);
		get_single_arg(val_arg, arg_g_cols, grid_total_cols);
		get_single_arg(val_arg, arg_rbm_n_epochs, rbm_n_epochs);
		get_single_arg(val_arg, arg_finetuning_n_epochs, finetuning_n_epochs);
		get_single_arg(val_arg, arg_n_samples, number_of_samples);
		get_single_arg(val_arg, arg_path_dataset, path_dataset);
		get_single_arg(val_arg, arg_batch_mode, batch_mode);
		get_layer_sizes(val_arg, arg_layers_sizes);
	}

	//il numero delle righe deve essere maggiore o uguale al numero di colonne
	//serve come presupposto per l'orientamento della griglia
	if(grid_total_rows < grid_total_cols)
	{
		//swap
		auto temp = grid_total_rows;
		grid_total_rows = {grid_total_cols};
		grid_total_cols = temp;
	}
}


void master_cout(string&& message){
	if(myid == 0)
		std::cout << message << "\n";
}

void set_generator(std::default_random_engine& generator)
{
	//todo capire se va bene utilizzando il proprio rand come seed
	srand(myid);

	generator.seed(myid);
}




void parallel_computation(std::ostream& oslog)
{
	master_cout("There are " + to_string(k_accumulators)
			+ " accumulators and the grid size is " + to_string(grid_total_rows) + "x" + to_string(grid_total_cols));


	//ottenimento generatore numeri casuali
	std::default_random_engine generator;
	set_generator(generator);

	oslog << "My first random number is " << generator() << "\n";


	//Determino comunicatori per ciascun nodo
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);


	//Ottenimento comunicatori
	master_cout("Computing comms");

	MPI_Comm master_acc_comm;
	my_vector<MP_Comm_MasterSlave> acc_row_comms;
	my_vector<MP_Comm_MasterSlave> acc_col_comms;

	GetCommsForMasterAcc(world_group, k_accumulators, master_acc_comm);
	GetCommsForGrid(world_group, GridOrientation::row_first, k_accumulators, grid_total_rows, grid_total_cols, acc_row_comms);
	GetCommsForGrid(world_group, GridOrientation::col_first, k_accumulators, grid_total_rows, grid_total_cols, acc_col_comms);


	//determino ruolo di ogni nodo
	master_cout("Instantiating classes");

	node_autoencoder* my_autoencoder;
	if(myid == 0)
	{
		//MASTER
		samples_manager smp_manager = samples_manager(path_dataset, number_of_samples);


		my_autoencoder = new node_master_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,
				oslog, myid,
				master_acc_comm,
				smp_manager);
	}
	else if(myid > 0 && (uint)myid <= k_accumulators)
	{
		//k accumulatori
		//indice del k-esimo accumulatore
		int k_number = myid - 1;

		my_autoencoder = new node_accumulator_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,
				oslog, myid,
				k_number,
				master_acc_comm, acc_row_comms, acc_col_comms);
	}
	else
	{
		//nodo giglia
		//determino l'n-esimo elemento della griglia
		uint grid_offset = myid - k_accumulators - 1;

		//quale riga?
		uint grid_row = grid_offset / grid_total_cols;

		//quale colonna?
		uint grid_col = grid_offset % grid_total_cols;

		my_autoencoder = new node_cell_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,
				oslog, myid,
				grid_row, grid_col,
				acc_row_comms, acc_col_comms);
	}


	MPI_Barrier(MPI_COMM_WORLD);
	master_cout("Begin loop");
	my_autoencoder->loop();
}


void single_computation(std::ostream& oslog)
{
	//MASTER
	samples_manager smp_manager = samples_manager(path_dataset, number_of_samples);
	std::default_random_engine generator;
	set_generator(generator);

	std::cout << "Running autoencoder on single node!\n";

	auto my_autoencoder = new node_single_autoencoder(layers_size, generator,
			rbm_n_epochs, finetuning_n_epochs, batch_mode,
			oslog,smp_manager);
	my_autoencoder->loop();
}


int main(int argc, char** argv) {

	//variabili temporali
    struct timeval wt1, wt0;
	double t0, t1;
	int numproc;

	//file di log
	std::filebuf fb;
	std::ostream oslog(&fb);

    try
    {
    	//get arguments
    	parse_args(argc, argv);

    	gettimeofday(&wt0, NULL);
		init_MPI(argc, argv, t0, myid, numproc);

		master_cout("Number of samples: " + to_string(number_of_samples));
		master_cout("Number of RBM training epochs: " + to_string(rbm_n_epochs));
		master_cout("Number of Fine-tuning training epoch: " + to_string(finetuning_n_epochs));
		master_cout("Batch mode: " + (batch_mode ? string("yes") : string("no")));
		master_cout("Path dataset: " + path_dataset);

		master_cout("Layer sizes: ");
		for(uint i = 0; i < layers_size.size();i++)
			master_cout(" - " + to_string(layers_size[i]) + " -");

		//se c'è un singolo nodo si esegue il codice in modalità non parallela
		bool parallel = numproc > 1;

		//creazione file
		string path_file = "logs/log_" + (parallel ? string("paral_") : string("single_")) + to_string(myid) + ".txt";
		fb.open(path_file, std::ios::out | std::ios::app);

		oslog << "---   Hello, I have ID " << myid << "\n";


		if(parallel)
			parallel_computation(oslog);
		else
			single_computation(oslog);


		//closing
		close_MPI(t1);
		gettimeofday(&wt1, NULL);

		//print results time
		//print_sec_mpi(std::cout, t0, t1, myid);
		//print_sec_mpi(oslog, t0, t1, myid);

		//print_sec_gtd(std::cout, wt0, wt1, myid);
		//print_sec_gtd(oslog, wt0, wt1, myid);

    }
    catch(const std::runtime_error& re)
    {
        // speciffic handling for runtime_error
        std::cerr << "Runtime error: " << re.what() << std::endl;
    }
    catch(const std::exception& ex)
    {
        // speciffic handling for all exceptions extending std::exception, except
        // std::runtime_error which is handled explicitly
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    }
    catch(...)
    {
        // catch any other errors (that we have no information about)
        std::cerr << "Unknown failure occurred. Possible memory corruption" << std::endl;
    }


    //chiusura file di log
    if(fb.is_open())
    {
    	master_cout("Closing log file");
		oslog.flush();
    	fb.close();
    }

    return 0;
}

