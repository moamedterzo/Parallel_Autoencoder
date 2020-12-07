/*
 * MPI_adapters.h
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#ifndef MPI_ADAPTERS_H_
#define MPI_ADAPTERS_H_


#include "mpi.h"
#include "custom_vectors.h"

namespace parallel_autoencoder
{
	//Data type to send for Autoencoder training
	const MPI_Datatype mpi_datatype_tosend = MPI_FLOAT;

	//utilizzata per identificare il processo root nell'insieme
	struct MPI_Comm_MasterSlave
	{
		MPI_Comm comm;
		uint acc_root_id;
		uint row_col_id;
		uint n_items_to_send;
	};



	inline void print_ssa(MPI_Status ssa[], int size = 1)
	{
		for(int i = 0;i< size; i++)

		{
			auto ss = ssa[i];
			std::string a = "Error: " + std::to_string(ss.MPI_ERROR) + ", " + std::to_string(ss.MPI_SOURCE) + ", " + std::to_string(ss.MPI_TAG) + "\n";
			std::cout << a;
		}
	}


	struct MPI_Req_Manager
	{
		MPI_Request *reqs;
		my_vector<MPI_Comm_MasterSlave> *comms;

		MPI_Req_Manager(MPI_Request *reqs, my_vector<MPI_Comm_MasterSlave> *comms)
		{
			this->comms = comms;
			this->reqs = reqs;
		}

		void wait()
		{
			MPI_Status ssa[comms->size()];
			//MPI_Waitall(comms->size(), reqs, MPI_STATUSES_IGNORE);
			MPI_Waitall(comms->size(), reqs, ssa);
			print_ssa(ssa, comms->size());
		}


		virtual ~MPI_Req_Manager(){}
	};



	struct MPI_Req_Manager_Cell : MPI_Req_Manager
	{
		MPI_Req_Manager_Cell(MPI_Request *reqs, my_vector<MPI_Comm_MasterSlave> *comms)
		: MPI_Req_Manager{reqs, comms }
		{	}


		void send_vector_to_reduce(my_vector<float>& vec)
		{
			//Send vector to accumulators
			int displacement = 0;
			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(vec.data() + displacement, MPI_IN_PLACE,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm , reqs + i);


				displacement += comm.n_items_to_send;
			}
		}


		void receive_vector(my_vector<float>& vec)
		{
			//Get vector from accumulators
			int displacement = 0;
			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement, comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm,  reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void receive_vector_sync(my_vector<float>& vec)
		{
			receive_vector(vec);
			wait();
		}

		~MPI_Req_Manager_Cell(){}
	};


	struct MPI_Req_Manager_Accumulator : MPI_Req_Manager
	{
		MPI_Req_Manager_Accumulator(MPI_Request *reqs, my_vector<MPI_Comm_MasterSlave> *comms)
		: MPI_Req_Manager{reqs, comms }
		{}


		void broadcast_vector(my_vector<float>& vec)
		{
			//Broadcast vector to cells
			int displacement = 0;
			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void accumulate_vector(my_vector<float>& vec)
		{
			int displacement = 0;

			//Accumulate vector summing all incoming vectors
			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(MPI_IN_PLACE, vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}


		void broadcast_vector_sync(my_vector<float>& vec)
		{
			broadcast_vector(vec);
			wait();
		}

		void accumulate_vector_sync(my_vector<float>& vec)
		{
			accumulate_vector(vec);
			wait();
		}



		~MPI_Req_Manager_Accumulator(){}
	};




	//Get comunicator for the master-accumulators group
	inline void get_comm_for_master_accs(const MPI_Group& world_group, const uint k_accumulators, MPI_Comm& master_acc_comm)
	{
		//Master and accumulators ranks
		int ranks_master_accs[1 + k_accumulators];
		for(uint i = 0; i != (1 + k_accumulators); i++)
			ranks_master_accs[i] = i;

		//create group
		MPI_Group master_acc_group;
		MPI_Group_incl(world_group, 1 + k_accumulators, ranks_master_accs, &master_acc_group);

		//create comm
		MPI_Comm_create_group(MPI_COMM_WORLD, master_acc_group, 0, &master_acc_comm);
	}



	//Get comunicators for accumulators-cells groups based on a specific orientation
	inline void get_comms_for_grid(const MPI_Group& world_group, const GridOrientation orientation, const uint k_accumulators,
			const uint grid_total_rows, const uint grid_total_cols, my_vector<MPI_Comm_MasterSlave>& acc_comms)
	{
		auto total_group_elements = orientation == GridOrientation::row_first ? grid_total_cols : grid_total_rows;
		auto total_groups_to_create = orientation == GridOrientation::row_first ? grid_total_rows : grid_total_cols;


		uint index_acc = 0, index_rowcol = 0;
		while(index_acc < k_accumulators || index_rowcol < total_groups_to_create)
		{
			//Getting ranks
			int acc_col_ranks[1 + total_group_elements];
			acc_col_ranks[0] = index_acc + 1; //rango accumulatore

			for(uint i = 0; i != total_group_elements; i++)
				//get ranks for row cells
				if(orientation == GridOrientation::row_first)
					acc_col_ranks[i + 1] = (k_accumulators + 1) + (index_rowcol * grid_total_cols) + i;
				//get ranks for colums cells
				else
					acc_col_ranks[i + 1] = (k_accumulators + 1) + i * grid_total_cols + index_rowcol;

			//create group from ranks
			MPI_Group acc_col_group;
			MPI_Group_incl(world_group, 1 + total_group_elements, acc_col_ranks, &acc_col_group);

			//Create struct for communicator
			MPI_Comm_MasterSlave acc_col_comm;
			MPI_Comm_create_group(MPI_COMM_WORLD, acc_col_group, 0, &acc_col_comm.comm);

			//add the communicator if it's not null
			if (MPI_COMM_NULL != acc_col_comm.comm) {

				acc_col_comm.acc_root_id = index_acc;
				acc_col_comm.row_col_id = index_rowcol;

				acc_comms.push_back(acc_col_comm);
			}

			//Compute difference in order to create the next group
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






	inline void init_MPI(int& argc, char** argv, int& myid, int& numprocs)
	{
		//init MPI and get time
		MPI_Init(&argc, &argv);

		MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	}


	inline void close_MPI()
	{
		MPI_Finalize();
	}
}



#endif /* MPI_ADAPTERS_H_ */
