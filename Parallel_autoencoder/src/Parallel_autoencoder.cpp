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


const int NUMBER_OF_SAMPLES = 8; //todo configurabile


//numero di elementi per layer predeterminato
vector<int> layers_size { 4096, 512, 256, 128, 64, 32 };

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
		int& k_accumulator, int& grid_rows, int& grid_cols)
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
		int k_accumulators, grid_total_rows, grid_total_cols;
		parse_args(argc, argv, k_accumulators, grid_total_rows, grid_total_cols);

		master_cout("There are " + to_string(k_accumulators)
				+ " accumulators and the grid size is " + to_string(grid_total_rows) + "x" + to_string(grid_total_cols));



		std::default_random_engine generator;
		set_generator(generator);

		oslog << "My first random number is " << generator() << "\n";



		//Determino comunicatori per ciascun nodo
		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);


		//comm accumulatori e master
		master_cout("Computing comm for master-acc");

		MP_Comm_MasterSlave master_acc_comm;
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

			//todo assicurarsi che il master abbia sempre rank 0
			//questo perché i master vengono sempre prima degli slave

			if (MPI_COMM_NULL != master_acc_comm.comm) {
				int prime_rank = -1;
				MPI_Comm_rank(master_acc_comm.comm, &prime_rank);
				oslog << "[World rank: "<< myid << ", my master_acc rank: " << prime_rank << "]\n\n";
			}
		}
		//fine ottenimento comm master acc


/*
		int sendCounts[5] = {4};  // everybody receives recvCount data
		sendCounts[0] = 0;                // but the root process
		int displs[5] = {0};
		for ( int i = 1; i < 5; i++ ) {
			displs[i] = (i - 1) * 4;
			sendCounts[i] = 4;
		}


		int n = 3000;
		vector<float> myvec(n), mysecvec(n);
					for(int i = 0; i < n;i++)
						myvec[i] = i * i;



					MPI_Request myreq, myreq2;
		if(myid == 0)
		{

			MPI_Isend(myvec.data(), myvec.size(), MPI_FLOAT, 1,
						0, master_acc_comm.comm, &myreq);

			MPI_Irecv(mysecvec.data(), mysecvec.size(), MPI_FLOAT, 1,
						0, master_acc_comm.comm, &myreq2);

			MPI_Wait(&myreq, MPI_STATUS_IGNORE);
			MPI_Wait(&myreq2, MPI_STATUS_IGNORE);

			std::cout << "im master\n";
			print_vector(mysecvec);

			for(auto c : sendCounts)
				std::cout << c << "\n";

			for(auto c : displs)
				std::cout << c << "\n";

			vector<float> myvec(16);
			for(int i = 0; i < 16;i++)
				myvec[i] = i * i;

			print_vector(myvec);

			int n_units_x_accumulator = 4;




			vector<float> mysinglevec(4);
			MPI_Request reqSend;

			MPI_Ireduce(MPI_IN_PLACE, mysinglevec.data(),
									4, MPI_FLOAT, MPI_SUM,
									0, master_acc_comm.comm ,  &reqSend);

			MPI_Iscatterv(myvec.data(), sendCounts, displs, MPI_FLOAT,
					nullptr, 0, MPI_FLOAT,
							0, master_acc_comm.comm, &reqSend);


			MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
			print_vector(mysinglevec);



		}
		else if(myid == 1)
		{
			MPI_Irecv(mysecvec.data(), mysecvec.size(), MPI_FLOAT, 0,
								0, master_acc_comm.comm, &myreq2);
			MPI_Isend(myvec.data(), myvec.size(), MPI_FLOAT, 0,
								0, master_acc_comm.comm, &myreq);


					MPI_Wait(&myreq, MPI_STATUS_IGNORE);
					MPI_Wait(&myreq2, MPI_STATUS_IGNORE);

					std::cout << "im acc\n";
					print_vector(mysecvec);

			//1/2/3/4
			vector<float> mysinglevec(4);
			for(int i = 0; i < 4;i++)
							mysinglevec[i] = i * i + myid;

			MPI_Request reqVis;



			MPI_Iscatterv(NULL, nullptr, nullptr, MPI_FLOAT,
					mysinglevec.data(), mysinglevec.size(), MPI_FLOAT,
				0, master_acc_comm.comm, &reqVis);

			MPI_Ireduce(mysinglevec.data(), MPI_IN_PLACE,
									4, MPI_FLOAT, MPI_SUM,
									0, master_acc_comm.comm ,&reqVis);



			MPI_Wait(&reqVis, MPI_STATUS_IGNORE);

		    std::this_thread::sleep_for(std::chrono::seconds(myid));
		    std::cout << "I'm number "<< myid << "\n";
			print_vector(mysinglevec);
		}


		  std::this_thread::sleep_for(std::chrono::seconds(100000));
		return 0;
*/










		//ottenimento comm accumulatori righe
		master_cout("Computing comm for acc-grid rows");
		vector<MP_Comm_MasterSlave> acc_row_comms;
		{
			int index_acc = 0, index_row = 0;
			while(index_acc < k_accumulators || index_row < grid_total_rows)
			{
				//creazione gruppo (un accumulatore più le celle della riga che corrispondono al numero di colonne)
				int acc_row_ranks[1 + grid_total_cols];
				acc_row_ranks[0] = index_acc + 1; //rango accumulatore
				//ranghi righe (k_accumulators + 1 va sommato perché rappresentano i ranghi assegnati al nodo master e a quelli accumulatori)
				for(int i = 0; i != grid_total_cols; i++)
					acc_row_ranks[i + 1] = (k_accumulators + 1) + (index_row * grid_total_cols) + i;

				MPI_Group acc_row_group;
				MPI_Group_incl(world_group, 1 + grid_total_cols, acc_row_ranks, &acc_row_group);

				MP_Comm_MasterSlave acc_row_comm;
				acc_row_comm.root_id = index_acc;
				acc_row_comm.row_col_id = index_row;
				MPI_Comm_create_group(MPI_COMM_WORLD, acc_row_group, 0, &acc_row_comm.comm);

				//se il processo corrente non fa parte del gruppo, non si aggiunge nulla
				if (MPI_COMM_NULL != acc_row_comm.comm) {
					acc_row_comms.push_back(acc_row_comm);

					//todo
					int prime_rank = -1;
					MPI_Comm_rank(acc_row_comm.comm, &prime_rank);
					oslog << "[World rank: "<< myid << ", my acc_row rank: " << prime_rank << "]\n\n";
					oslog.flush();
				}

				//questa differenza serve per generare le associazioni tra righe e accumulatori
				int diff = (index_acc + 1) * grid_total_rows - (index_row + 1) * k_accumulators;

				if(diff == 0)
				{
					index_acc++;
					index_row++;
				}
				else if(diff > 0)
				{
					index_row++;
				}
				else
				{
					index_acc++;
				}
			}
		}
		//fine ottenimento acc righe

		//ottenimento comm accumulatori colonne
		master_cout("Computing comm for acc-grid cols");

		vector<MP_Comm_MasterSlave> acc_col_comms;
		{
			int index_acc = 0, index_col = 0;
			while(index_acc < k_accumulators || index_col < grid_total_cols)
			{
				//creazione gruppo (un accumulatore più le celle della riga che corrispondono al numero di colonne)
				int acc_col_ranks[1 + grid_total_rows];
				acc_col_ranks[0] = index_acc + 1; //rango accumulatore
				//ranghi colonne (k_accumulators + 1 va sommato perché rappresentano i ranghi assegnati al nodo master e a quelli accumulatori)
				for(int i = 0; i != grid_total_rows; i++)
					acc_col_ranks[i + 1] = (k_accumulators + 1) + i * grid_total_cols + index_col;

				MPI_Group acc_col_group;
				MPI_Group_incl(world_group, 1 + grid_total_rows, acc_col_ranks, &acc_col_group);

				MP_Comm_MasterSlave acc_col_comm;
				acc_col_comm.root_id = index_acc;
				acc_col_comm.row_col_id = index_col;
				MPI_Comm_create_group(MPI_COMM_WORLD, acc_col_group, 0, &acc_col_comm.comm);

				//se il processo corrente non fa parte del gruppo, non si aggiunge nulla
				if (MPI_COMM_NULL != acc_col_comm.comm) {
					acc_col_comms.push_back(acc_col_comm);

					//todo
					int prime_rank = -1;
					MPI_Comm_rank(acc_col_comm.comm, &prime_rank);
					oslog << "[World rank: "<< myid << ", my acc_col rank: " << prime_rank << "]\n\n";
					oslog.flush();
				}

				//questa differenza serve per generare le associazioni tra righe e accumulatori
				int diff = (index_acc + 1) * grid_total_cols - (index_col + 1) * k_accumulators;

				if(diff == 0)
				{
					index_acc++;
					index_col++;
				}
				else if(diff > 0)
				{
					index_col++;
				}
				else
				{
					index_acc++;
				}
			}
		}
		//fine ottenimento acc colonne



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
			int grid_offset = myid - k_accumulators - 1;

			//quale riga?
			int grid_row = grid_offset / grid_total_cols;

			//quale colonna?
			int grid_col = grid_offset % grid_total_cols;

			my_autoencoder = new node_cell_autoencoder(layers_size, generator, k_accumulators,
					grid_total_rows, grid_total_cols,
					oslog, myid,
					grid_row, grid_col,
					acc_row_comms, acc_col_comms);
		}

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

