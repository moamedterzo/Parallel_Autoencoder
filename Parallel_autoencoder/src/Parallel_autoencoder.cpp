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
my_vector<int> layers_size { 4096, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32};

uint k_accumulators = 2, grid_total_rows = 2, grid_total_cols = 2;

string path_dataset = "./mnist_chinese/data";
uint number_of_samples = 2;
uint rbm_n_epochs = 80;
uint finetuning_n_epochs = 5;
bool batch_mode = false;
bool reduce_io = false;
bool execute_command = false;
CommandType commandToExecute;


//rank MPI id
int mpi_my_rank;


//METODI PER L'OTTENMENTO DEGLI ARGOMENTI IN INPUT

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

void get_command(std::string& s, const std::string& arg_name)
{
	std::size_t pos = s.find(arg_name);

	if (pos != std::string::npos)
	{
		execute_command = true;
		commandToExecute = static_cast<CommandType>(std::stoi(s.replace(pos, arg_name.length(), "")));
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
	static const string arg_reduce_io = "--reduce_io:";
	static const string arg_n_samples = "--n_samples:";
	static const string arg_path_dataset = "--path_dataset:";
	static const string arg_layers_sizes = "--layer_sizes:";
	static const string arg_command = "--command:";

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
		get_single_arg(val_arg, arg_reduce_io, reduce_io);
		get_layer_sizes(val_arg, arg_layers_sizes);
		get_command(val_arg, arg_command);
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

//FINE METODI PER L'OTTENMENTO DEGLI ARGOMENTI IN INPUT


//Solamente il master mostra a video un messagge
void master_cout(string&& message){
	if(mpi_my_rank == 0)
		std::cout << message << "\n";
}

void set_generator(std::default_random_engine& generator)
{
	//Come seed viene utilizzato il rango di MPI
	srand(mpi_my_rank);
	generator.seed(mpi_my_rank);
}



//Avvia i processi per la computazione parallela
void parallel_computation(std::ostream& oslog)
{
	master_cout("There are " + to_string(k_accumulators)
			+ " accumulators and the grid size is " + to_string(grid_total_rows) + "x" + to_string(grid_total_cols));


	//ottenimento generatore numeri casuali
	std::default_random_engine generator;
	set_generator(generator);
	oslog << "My first random number is " << generator() << "\n";


	//Determino comunicatori per ciascun nodo
	master_cout("Computing communicators...");

	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	MPI_Comm master_acc_comm;
	my_vector<MPI_Comm_MasterSlave> acc_row_comms;
	my_vector<MPI_Comm_MasterSlave> acc_col_comms;

	get_comm_for_master_accs(world_group, k_accumulators, master_acc_comm);
	get_comms_for_grid(world_group, GridOrientation::row_first, k_accumulators, grid_total_rows, grid_total_cols, acc_row_comms);
	get_comms_for_grid(world_group, GridOrientation::col_first, k_accumulators, grid_total_rows, grid_total_cols, acc_col_comms);

	//todo rimuovere


	if(master_acc_comm != MPI_COMM_NULL)
		MPI_Errhandler_set(master_acc_comm, MPI_ERRORS_RETURN);
	//MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	if((uint)mpi_my_rank > k_accumulators)
	{
		for(int i = 0; i < acc_row_comms.size(); i++)
				MPI_Errhandler_set(acc_row_comms[i].comm, MPI_ERRORS_RETURN);
		for(int i = 0; i < acc_col_comms.size(); i++)
				MPI_Errhandler_set(acc_col_comms[i].comm, MPI_ERRORS_RETURN);
	}

	//determino ruolo di ogni nodo
	master_cout("Initializing objects...");

	node_autoencoder* my_autoencoder;
	if(mpi_my_rank == 0)
	{
		//MASTER
		samples_manager smp_manager = samples_manager(path_dataset, number_of_samples);


		my_autoencoder = new node_master_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,reduce_io,
				oslog, mpi_my_rank,
				master_acc_comm,
				smp_manager);
	}
	else if(mpi_my_rank > 0 && (uint)mpi_my_rank <= k_accumulators)
	{
		//ACCUMULATORE
		//indice del k-esimo accumulatore
		int k_number = mpi_my_rank - 1;

		my_autoencoder = new node_accumulator_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,reduce_io,
				oslog, mpi_my_rank,
				k_number,
				master_acc_comm, acc_row_comms, acc_col_comms);
	}
	else
	{
		//CELLA
		uint grid_offset = mpi_my_rank - k_accumulators - 1;

		//quale riga e quale colonna?
		uint grid_row = grid_offset / grid_total_cols;
		uint grid_col = grid_offset % grid_total_cols;

		my_autoencoder = new node_cell_autoencoder(layers_size, generator, k_accumulators,
				grid_total_rows, grid_total_cols,
				rbm_n_epochs, finetuning_n_epochs, batch_mode,reduce_io,
				oslog, mpi_my_rank,
				grid_row, grid_col,
				acc_row_comms, acc_col_comms);
	}

	//esecuzione comando
	if(execute_command)
		my_autoencoder->execute_command(commandToExecute);
	else
		my_autoencoder->loop();
}


//Computazione su un singolo nodo
void single_computation(std::ostream& oslog)
{
	//SINGLE MASTER
	samples_manager smp_manager = samples_manager(path_dataset, number_of_samples);
	std::default_random_engine generator;
	set_generator(generator);

	std::cout << "\n\nRunning autoencoder on single node!\n";

	auto my_autoencoder = new node_single_autoencoder(layers_size, generator,
			rbm_n_epochs, finetuning_n_epochs, batch_mode, reduce_io,
			oslog,smp_manager);

	if(execute_command)
		my_autoencoder->execute_command(commandToExecute);
	else
		my_autoencoder->loop();
}


int main(int argc, char** argv) {

	//file di log
	std::filebuf fb;
	std::ostream oslog(&fb);

	//get arguments
	parse_args(argc, argv);

	//init MPI
	int numproc;
	init_MPI(argc, argv, mpi_my_rank, numproc);

	//print variabiles
	master_cout("Number of samples: " + to_string(number_of_samples));
	master_cout("Number of RBM training epochs: " + to_string(rbm_n_epochs));
	master_cout("Number of Fine-tuning training epoch: " + to_string(finetuning_n_epochs));
	master_cout("Batch mode: " + (batch_mode ? string("yes") : string("no")));
	master_cout("Reduce IO: " + (reduce_io ? string("yes") : string("no")));
	master_cout("Path dataset: " + path_dataset);

	master_cout("Layer sizes: ");
	for(uint i = 0; i < layers_size.size();i++)
		master_cout(" - " + to_string(layers_size[i]) + " -");


	//se c'è un singolo nodo si esegue il codice in modalità non parallela
	bool parallel = numproc > 1;

	//creazione file per ciascun nodo
	string path_file = "logs/log_" + (parallel ? string("paral_") : string("single_")) + to_string(mpi_my_rank) + ".txt";
	fb.open(path_file, std::ios::out | std::ios::app);

	oslog << "---   Hello, I have ID " << mpi_my_rank << "\n";

	//computazione
	try
	{
		if(parallel)
			parallel_computation(oslog);
		else
			single_computation(oslog);
	}
	catch(MPI::Exception& ex)
	{
		std::cout << "Error MPI from rank " + to_string(mpi_my_rank) + "\n";
		std::cout  << ex.Get_error_class() << "\n";
		std::cout  << ex.Get_error_string() << "\n";
		std::cout  << ex.Get_error_code() << "\n";
	}
	catch(...)
	{
		std::cout << "Error from rank " + to_string(mpi_my_rank) + "\n";
	}



	//closing MPI
	close_MPI();

    //chiusura file di log
    if(fb.is_open())
    {
    	master_cout("Closing log file");
		oslog.flush();
    	fb.close();
    }

    return 0;
}

