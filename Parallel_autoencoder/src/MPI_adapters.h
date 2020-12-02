/*
 * MPI_adapters.h
 *
 *  Created on: 30 nov 2020
 *      Author: giovanni
 */

#ifndef MPI_ADAPTERS_H_
#define MPI_ADAPTERS_H_


#include <mpi/mpi.h>
#include "custom_vectors.h"

namespace parallel_autoencoder
{
	const MPI_Datatype mpi_datatype_tosend = MPI_FLOAT;

	//utilizzata per identificare il processo root nell'insieme
	struct MP_Comm_MasterSlave{

		MPI_Comm comm;
		uint acc_root_id;
		uint row_col_id;
		uint n_items_to_send;
	};


	struct MPReqManager
	{
		MPReqManager(MPI_Request *reqs,my_vector<MP_Comm_MasterSlave> *comms)
		{
			this->comms = comms;
			this->reqs = reqs;
		}

		MPI_Request *reqs;
		my_vector<MP_Comm_MasterSlave> *comms;


		void wait()
		{
			MPI_Waitall(comms->size(), reqs, MPI_STATUSES_IGNORE);
		}


		virtual ~MPReqManager(){}
	};


	struct MPReqManagerCell : MPReqManager
	{
		MPReqManagerCell(MPI_Request *reqs, my_vector<MP_Comm_MasterSlave> *comms)
		: MPReqManager{reqs, comms }
		{}


		void SendVectorToReduce(my_vector<float>& vec)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(vec.data() + displacement, MPI_IN_PLACE,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm , reqs + i);


				displacement += comm.n_items_to_send;
			}
		}


		void ReceiveVector(my_vector<float>& vec)
		{
			//Invio vettore agli accumululatori di riferimento
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement, comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm,  reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void ReceiveVectorSync(my_vector<float>& vec)
		{
			ReceiveVector(vec);
			wait();
		}

		~MPReqManagerCell(){}
	};


	struct MPReqManagerAccumulator : MPReqManager
	{
		MPReqManagerAccumulator(MPI_Request *reqs, my_vector<MP_Comm_MasterSlave> *comms)
		: MPReqManager{reqs, comms }
		{}


		void BroadcastVector(my_vector<float>& vec)
		{
			//Si invia il vettore alle righe/colonne di riferimento, per ognuna si usa il broadcast
			int displacement = 0;
			for(uint i = 0; i < comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ibcast(vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}

		void AccumulateVector(my_vector<float>& vec)
		{
			int displacement = 0;

			for(uint i = 0; i != comms->size(); i++)
			{
				auto& comm = (*comms)[i];

				MPI_Ireduce(MPI_IN_PLACE, vec.data() + displacement,
						comm.n_items_to_send, mpi_datatype_tosend, MPI_SUM,
						0, comm.comm, reqs + i);

				displacement += comm.n_items_to_send;
			}
		}


		void BroadcastVectorSync(my_vector<float>& vec)
		{
			BroadcastVector(vec);
			wait();
		}

		void AccumulateVectorSync(my_vector<float>& vec)
		{
			AccumulateVector(vec);
			wait();
		}



		~MPReqManagerAccumulator(){}
	};





	inline void GetCommsForMasterAcc(MPI_Group& world_group, const uint k_accumulators, MPI_Comm& master_acc_comm)
	{
		//insieme di rank di master e accumulatori
		//va da 0 a K
		int ranks_master_accs[1 + k_accumulators];
		for(uint i = 0; i != (1 + k_accumulators); i++)
			ranks_master_accs[i] = i;

		//gruppo e comunicatore master accumulatori
		MPI_Group master_acc_group;
		MPI_Group_incl(world_group, 1 + k_accumulators, ranks_master_accs, &master_acc_group);

		MPI_Comm_create_group(MPI_COMM_WORLD, master_acc_group, 0, &master_acc_comm);

	}


	inline void GetCommsForGrid(MPI_Group& world_group, GridOrientation orientation, const uint k_accumulators,
			const uint grid_total_rows, const uint grid_total_cols, my_vector<MP_Comm_MasterSlave>& acc_comms)
	{
		auto total_group_elements = orientation == GridOrientation::row_first ? grid_total_cols : grid_total_rows;
		auto total_groups_to_create = orientation == GridOrientation::row_first ? grid_total_rows : grid_total_cols;


		uint index_acc = 0, index_rowcol = 0;
		while(index_acc < k_accumulators || index_rowcol < total_groups_to_create)
		{
			//creazione gruppo (un accumulatore più le celle della riga che corrispondono al numero di colonne)
			int acc_col_ranks[1 + total_group_elements];
			acc_col_ranks[0] = index_acc + 1; //rango accumulatore

			//ranghi colonne (k_accumulators + 1 va sommato perché rappresentano i ranghi assegnati al nodo master e a quelli accumulatori)
			for(uint i = 0; i != total_group_elements; i++)
				if(orientation == GridOrientation::row_first)
					acc_col_ranks[i + 1] = (k_accumulators + 1) + (index_rowcol * grid_total_cols) + i;
				else
					acc_col_ranks[i + 1] = (k_accumulators + 1) + i * grid_total_cols + index_rowcol;


			MPI_Group acc_col_group;
			MPI_Group_incl(world_group, 1 + total_group_elements, acc_col_ranks, &acc_col_group);

			MP_Comm_MasterSlave acc_col_comm;
			acc_col_comm.acc_root_id = index_acc;
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






	inline void init_MPI(int& argc, char** argv,
			double& t0,	int& myid, int& numprocs)
	{
		//init MPI e suoi tempi
		MPI_Init(&argc, &argv);
		t0 = MPI_Wtime();

		MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	}


	inline void close_MPI(double& t1)
	{
		t1 = MPI_Wtime();
		MPI_Finalize();
	}



}



#endif /* MPI_ADAPTERS_H_ */
