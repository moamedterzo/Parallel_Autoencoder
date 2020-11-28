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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include "mpi.h"
#include <sys/time.h>
#include <random>
#include <string>

#include <thread>         // std::this_thread::sleep_for eliminare
#include <chrono>         // std::chrono::seconds eliminare


#include "custom_utils.h"
#include "samples_manager.h"
#include "autoencoder.h"


#include "node_autoencoder.h"
#include "node_accumulator_autoencoder.h"
#include "node_master_autoencoder.h"
#include "node_cell_autoencoder.h"

using namespace std;
using namespace parallel_autoencoder;


const int NUMBER_OF_SAMPLES = 2; //todo configurabile


//numero di elementi per layer predeterminato
my_vector<int> layers_size { 4096, 8, 16, 32, 64, 32 };

const string PATH_DATASET = "./mnist_chinese/data";


int myid;
int numproc;


double diffmsec(const struct timeval & a,
                                    const struct timeval & b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);

    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}



void init_MPI(int& argc, char** argv,
		timeval& wt0, double& t0,
		int& myid, int& numprocs)
{
	gettimeofday(&wt0, NULL);

	//int namelen;
	//char processor_name[MPI_MAX_PROCESSOR_NAME];

	//init MPI e suoi tempi
	MPI_Init(&argc, &argv);
	t0 = MPI_Wtime();


	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	//MPI_Get_processor_name(processor_name, &namelen);

	 MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

}

void close_MPI(timeval& wt1, const timeval& wt0,
		double& t1, const double& t0,
		const int& myid)
{
	//close MPI + tempi finali
	t1 = MPI_Wtime();
	MPI_Finalize();
	gettimeofday(&wt1, NULL);

	//statistiche tempi
	auto diffsec = diffmsec(wt1, wt0)/1000 ;

	std::cout << "total time (MPI) " << myid << " is " << t1 - t0 << "\n";
	std::cout << "total time (gtd) " << myid << " is " << diffsec << "\n";
}


void parse_args(const int argc, char** argv,
		uint& k_accumulator, uint& grid_rows, uint& grid_cols)
{
	k_accumulator = 4;
	grid_rows = 4;
	grid_cols = 2;

	//todo leggere parametri in input


	//il numero delle righe deve essere maggiore o uguale al numero di colonne
	//serve come presupposto per l'orientamento della griglia
	if(grid_rows < grid_cols)
	{
		//swap
		auto temp = grid_rows;
		grid_rows = grid_cols;
		grid_cols = temp;
	}
}

void master_cout(string&& message)
{
	if(myid == 0)
	{
		std::cout << message << "\n";
	}
}

void set_generator(std::default_random_engine& generator)
{
	//todo capire se va bene utilizzando il proprio rand come seed
	srand(myid);

	generator.seed(myid);
}



void GetCommsForMasterAcc(MPI_Group& world_group, const uint k_accumulators, MP_Comm_MasterSlave& master_acc_comm)
{
	master_acc_comm.root_id = 0;
	{
		//insieme di rank di master e accumulatori
		//va da 0 a K
		int ranks_master_accs[1 + k_accumulators];
		for(int i = 0; i != (1 + k_accumulators); i++)
			ranks_master_accs[i] = i;

		//gruppo e comunicatore master accumulatori
		MPI_Group master_acc_group;
		MPI_Group_incl(world_group, 1 + k_accumulators, ranks_master_accs, &master_acc_group);

		MPI_Comm_create_group(MPI_COMM_WORLD, master_acc_group, 0, &master_acc_comm.comm);
	}
}


void GetCommsForGrid(MPI_Group& world_group, GridOrientation orientation, const uint k_accumulators,
		const uint grid_total_rows, const uint grid_total_cols, my_vector<MP_Comm_MasterSlave>& acc_comms)
{
	auto total_group_elements = orientation == row_first ? grid_total_cols : grid_total_rows;
	auto total_groups_to_create = orientation == row_first ? grid_total_rows : grid_total_cols;


	uint index_acc = 0, index_rowcol = 0;
	while(index_acc < k_accumulators || index_rowcol < total_groups_to_create)
	{
		//creazione gruppo (un accumulatore più le celle della riga che corrispondono al numero di colonne)
		int acc_col_ranks[1 + total_group_elements];
		acc_col_ranks[0] = index_acc + 1; //rango accumulatore

		//ranghi colonne (k_accumulators + 1 va sommato perché rappresentano i ranghi assegnati al nodo master e a quelli accumulatori)
		for(uint i = 0; i != total_group_elements; i++)
			if(orientation == row_first)
				acc_col_ranks[i + 1] = (k_accumulators + 1) + (index_rowcol * grid_total_cols) + i;
			else
				acc_col_ranks[i + 1] = (k_accumulators + 1) + i * grid_total_cols + index_rowcol;


		MPI_Group acc_col_group;
		MPI_Group_incl(world_group, 1 + total_group_elements, acc_col_ranks, &acc_col_group);

		MP_Comm_MasterSlave acc_col_comm;
		acc_col_comm.root_id = index_acc;
		acc_col_comm.row_col_id = index_rowcol;
		MPI_Comm_create_group(MPI_COMM_WORLD, acc_col_group, 0, &acc_col_comm.comm);

		//se il processo corrente non fa parte del gruppo, non si aggiunge nulla
		if (MPI_COMM_NULL != acc_col_comm.comm) {
			acc_comms.push_back(acc_col_comm);
		}

		//questa differenza serve per generare le associazioni tra righe e accumulatori
		int diff = (index_acc + 1) * total_groups_to_create - (index_rowcol + 1) * k_accumulators;

		if(diff == 0)
		{
			index_acc++;
			index_rowcol++;
		}
		else if(diff > 0)
		{
			index_rowcol++;
		}
		else
		{
			index_acc++;
		}
	}
}




int main(int argc, char** argv) {

    struct timeval wt1, wt0;
	double t0, t1;

	//file di log
	std::filebuf fb;
	std::ostream oslog(&fb);

    try
    {
		init_MPI(argc, argv, wt0, t0, myid, numproc);


		//creazione file
		string path_file = "logs/log_"+ to_string(myid) + ".txt";
		fb.open(path_file, std::ios::out | std::ios::app);

		oslog << "---   Hello, I have ID " << myid << "\n";


		//leggo configurazione griglia
		uint k_accumulators, grid_total_rows, grid_total_cols;
		parse_args(argc, argv, k_accumulators, grid_total_rows, grid_total_cols);

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

		MP_Comm_MasterSlave master_acc_comm;
		my_vector<MP_Comm_MasterSlave> acc_row_comms;
		my_vector<MP_Comm_MasterSlave> acc_col_comms;

		GetCommsForMasterAcc(world_group, k_accumulators, master_acc_comm);
		GetCommsForGrid(world_group, row_first, k_accumulators, grid_total_rows, grid_total_cols, acc_row_comms);
		GetCommsForGrid(world_group, col_first, k_accumulators, grid_total_rows, grid_total_cols, acc_col_comms);


		//determino ruolo di ogni nodo
		master_cout("Instantiating classes");

		node_autoencoder* my_autoencoder;
		if(myid == 0)
		{
			//MASTER
			samples_manager smp_manager = samples_manager(PATH_DATASET, NUMBER_OF_SAMPLES);


			my_autoencoder = new node_master_autoencoder(layers_size, generator, k_accumulators,
					grid_total_rows, grid_total_cols,
					oslog, myid,
					master_acc_comm,
					smp_manager);
		}
		else if(myid > 0 && myid <= k_accumulators)
		{
			//k accumulatori
			//indice del k-esimo accumulatore
			int k_number = myid - 1;

			my_autoencoder = new node_accumulator_autoencoder(layers_size, generator, k_accumulators,
					grid_total_rows, grid_total_cols,
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
					oslog, myid,
					grid_row, grid_col,
					acc_row_comms, acc_col_comms);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		master_cout("Begin loop");
		my_autoencoder->loop();


/*

        std::default_random_engine generator;
        samples_manager sss = samples_manager("./mnist_chinese/data", 4); //todo sistemare

        vector<int> layer_sizes = { 4096 , 1024,  512 , 256, 128, 64, 32};

        Autoencoder autoen = Autoencoder(layer_sizes, sss, generator);
        //autoen.load_parameters();

        autoen.Train();


        string path_to_save = "./autoencoder_pars/saved_pars.txt";
        autoen.save_parameters(path_to_save);


        //proviamo il reconstruct
        sss.restart();
        vector<float> input_buffer(layer_sizes[0]);
        while(sss.get_next_sample(input_buffer)){

            auto reconstructed = autoen.reconstruct(input_buffer);

            std::cout << "Root squared error: " << root_squared_error(input_buffer, reconstructed) << "\n";

            std::cout << "Input vector\n";
            //print_vector(input_buffer);
            sss.show_sample(input_buffer);

            std::cout << "Output vector\n";
            //print_vector(reconstructed);
            sss.show_sample(reconstructed);

            //getchar();
        }
*/

    	close_MPI(wt1, wt0, t1, t0, myid);
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

