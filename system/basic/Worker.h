#ifndef WORKER_H
#define WORKER_H

#include <iomanip>
#include <vector>
#include "../utils/global.h"
#include "MessageBuffer.h"
#include "metrics_logger.h"
#include <string>
#include "../utils/communication.h"
#include "../utils/ydhdfs.h"
#include "../utils/Combiner.h"
#include "../utils/Aggregator.h"
using namespace std;

template <class VertexT, class AggregatorT = DummyAgg> //user-defined VertexT
class Worker {
    typedef vector<VertexT*> VertexContainer;
    typedef typename VertexContainer::iterator VertexIter;

    typedef typename VertexT::KeyType KeyT;
    typedef typename VertexT::MessageType MessageT;
    typedef typename VertexT::HashType HashT;

    typedef MessageBuffer<VertexT> MessageBufT;
    typedef typename MessageBufT::MessageContainerT MessageContainerT;
    typedef typename MessageBufT::Map Map;
    typedef typename MessageBufT::MapIter MapIter;

    typedef typename AggregatorT::PartialType PartialT;
    typedef typename AggregatorT::FinalType FinalT;

public:
    Worker()
    {
        //init_workers();//put to run.cpp
        message_buffer = new MessageBuffer<VertexT>;
        global_message_buffer = message_buffer;
        active_count = 0;
        combiner = NULL;
        global_combiner = NULL;
        aggregator = NULL;
        global_aggregator = NULL;
        global_agg = NULL;
    }

    void setCombiner(Combiner<MessageT>* cb)
    {
        combiner = cb;
        global_combiner = cb;
    }

    void setAggregator(AggregatorT* ag)
    {
        aggregator = ag;
        global_aggregator = ag;
        global_agg = new FinalT;
    }

    virtual ~Worker()
    {
        for (int i = 0; i < vertexes.size(); i++)
            delete vertexes[i];
        delete message_buffer;
        if (getAgg() != NULL)
            delete (FinalT*)global_agg;
        //worker_finalize();//put to run.cpp
        worker_barrier(); //newly added for ease of multi-job programming in run.cpp
    }

    //==================================
    //sub-functions
    void sync_graph()
    {
        //ResetTimer(4);
        //set send buffer
        vector<VertexContainer> _loaded_parts(_num_workers);
        for (int i = 0; i < vertexes.size(); i++) {
            VertexT* v = vertexes[i];
            _loaded_parts[hash(v->id)].push_back(v);
        }
        //exchange vertices to add
        all_to_all(_loaded_parts);

        //delete sent vertices
        for (int i = 0; i < vertexes.size(); i++) {
            VertexT* v = vertexes[i];
            if (hash(v->id) != _my_rank)
                delete v;
        }
        vertexes.clear();
        //collect vertices to add
        for (int i = 0; i < _num_workers; i++) {
            vertexes.insert(vertexes.end(), _loaded_parts[i].begin(), _loaded_parts[i].end());
        }
        _loaded_parts.clear();
        //StopTimer(4);
        //PrintTimer("Reduce Time",4);
    };

    //old implementation
    /*
		void active_compute()
		{
			active_count=0;
			MessageBufT* mbuf=(MessageBufT*)get_message_buffer();
			Map & msgs=mbuf->get_messages();
			MessageContainerT empty;
			for(VertexIter it=vertexes.begin(); it!=vertexes.end(); it++)
			{
				KeyT vid=(*it)->id;
				MapIter mit=msgs.find(vid);
				if(mit->second->size()==0)
				{
					if((*it)->is_active())
					{
						(*it)->compute(empty);
						AggregatorT* agg=(AggregatorT*)get_aggregator();
						if(agg!=NULL) agg->stepPartial(*it);
						if((*it)->is_active()) active_count++;
					}
				}
				else
				{
					(*it)->activate();
					(*it)->compute(*(mit->second));
					mit->second->clear();//clear used msgs
					AggregatorT* agg=(AggregatorT*)get_aggregator();
					if(agg!=NULL) agg->stepPartial(*it);
					if((*it)->is_active()) active_count++;
				}
			}
		}

		void all_compute()
		{
			active_count=0;
			MessageBufT* mbuf=(MessageBufT*)get_message_buffer();
			Map & msgs=mbuf->get_messages();
			MessageContainerT empty;
			for(VertexIter it=vertexes.begin(); it!=vertexes.end(); it++)
			{
				KeyT vid=(*it)->id;
				MapIter mit=msgs.find(vid);
				(*it)->activate();
				if(mit->second->size()==0) (*it)->compute(empty);
				else{
					(*it)->compute(*(mit->second));
					mit->second->clear();//clear used msgs
				}
				AggregatorT* agg=(AggregatorT*)get_aggregator();
				if(agg!=NULL) agg->stepPartial(*it);
				if((*it)->is_active()) active_count++;
			}
		}
		*/

    void active_compute()
    {
        active_count = 0;
        MessageBufT* mbuf = (MessageBufT*)get_message_buffer();
        vector<MessageContainerT>& v_msgbufs = mbuf->get_v_msg_bufs();
        for (int i = 0; i < vertexes.size(); i++) {
            if (v_msgbufs[i].size() == 0) {
                if (vertexes[i]->is_active()) {
                    vertexes[i]->compute(v_msgbufs[i]);
                    AggregatorT* agg = (AggregatorT*)get_aggregator();
                    if (agg != NULL)
                        agg->stepPartial(vertexes[i]);
                    if (vertexes[i]->is_active())
                        active_count++;
                }
            } else {
                vertexes[i]->activate();
                vertexes[i]->compute(v_msgbufs[i]);
                v_msgbufs[i].clear(); //clear used msgs
                AggregatorT* agg = (AggregatorT*)get_aggregator();
                if (agg != NULL)
                    agg->stepPartial(vertexes[i]);
                if (vertexes[i]->is_active())
                    active_count++;
            }
        }
    }

    void all_compute()
    {
        active_count = 0;
        MessageBufT* mbuf = (MessageBufT*)get_message_buffer();
        vector<MessageContainerT>& v_msgbufs = mbuf->get_v_msg_bufs();
        for (int i = 0; i < vertexes.size(); i++) {
            vertexes[i]->activate();
            vertexes[i]->compute(v_msgbufs[i]);
            v_msgbufs[i].clear(); //clear used msgs
            AggregatorT* agg = (AggregatorT*)get_aggregator();
            if (agg != NULL)
                agg->stepPartial(vertexes[i]);
            if (vertexes[i]->is_active())
                active_count++;
        }
    }

    inline void add_vertex(VertexT* vertex)
    {
        vertexes.push_back(vertex);
        if (vertex->is_active())
            active_count++;
    }

    void agg_sync()
    {
        AggregatorT* agg = (AggregatorT*)get_aggregator();
        if (agg != NULL) {
            if (_my_rank != MASTER_RANK) { //send partialT to aggregator
                //gathering PartialT
                PartialT* part = agg->finishPartial();
                //------------------------ strategy choosing BEGIN ------------------------
                StartTimer(COMMUNICATION_TIMER);
                StartTimer(SERIALIZATION_TIMER);
                ibinstream m;
                m << part;
                int sendcount = m.size();
                StopTimer(SERIALIZATION_TIMER);
                int total = all_sum(sendcount);
                StopTimer(COMMUNICATION_TIMER);
                //------------------------ strategy choosing END ------------------------
                if (total <= AGGSWITCH)
                    slaveGather(*part);
                else {
                    send_ibinstream(m, MASTER_RANK);
                }
                //scattering FinalT
                slaveBcast(*((FinalT*)global_agg));
            } else {
                //------------------------ strategy choosing BEGIN ------------------------
                int total = all_sum(0);
                //------------------------ strategy choosing END ------------------------
                //gathering PartialT
                if (total <= AGGSWITCH) {
                    vector<PartialT*> parts(_num_workers);
                    masterGather(parts);
                    for (int i = 0; i < _num_workers; i++) {
                        if (i != MASTER_RANK) {
                            PartialT* part = parts[i];
                            agg->stepFinal(part);
                            delete part;
                        }
                    }
                } else {
                    for (int i = 0; i < _num_workers; i++) {
                        if (i != MASTER_RANK) {
                            obinstream um = recv_obinstream(i);
                            PartialT* part;
                            um >> part;
                            agg->stepFinal(part);
                            delete part;
                        }
                    }
                }
                //scattering FinalT
                FinalT* final = agg->finishFinal();
                //cannot set "global_agg=final" since MASTER_RANK works as a slave, and agg->finishFinal() may change
                *((FinalT*)global_agg) = *final; //deep copy
                masterBcast(*((FinalT*)global_agg));
            }
        }
    }

    //user-defined graphLoader ==============================
    virtual VertexT* toVertex(char* line) = 0; //this is what user specifies!!!!!!

    void load_vertex(VertexT* v)
    { //called by load_graph
        add_vertex(v);
    }

    void load_graph(const char* inpath)
    {
        hdfsFS fs = getHdfsFS();
        hdfsFile in = getRHandle(inpath, fs);
        LineReader reader(fs, in);
        while (true) {
            reader.readLine();
            if (!reader.eof())
                load_vertex(toVertex(reader.getLine()));
            else
                break;
        }
        hdfsCloseFile(fs, in);
        hdfsDisconnect(fs);
        //cout<<"Worker "<<_my_rank<<": \""<<inpath<<"\" loaded"<<endl;//DEBUG !!!!!!!!!!
    }
    //=======================================================

    //user-defined graphDumper ==============================
    virtual void toline(VertexT* v, BufferedWriter& writer) = 0; //this is what user specifies!!!!!!

    void dump_partition(const char* outpath)
    {
        hdfsFS fs = getHdfsFS();
        BufferedWriter* writer = new BufferedWriter(outpath, fs, _my_rank);

        for (VertexIter it = vertexes.begin(); it != vertexes.end(); it++) {
            writer->check();
            toline(*it, *writer);
        }
        delete writer;
        hdfsDisconnect(fs);
    }
    //=======================================================

    // run the worker
    // --- load: graph loading + partitioning phase (call once) ---
    void load(const WorkerParams& params)
    {
        init_timers();
        ResetTimer(WORKER_TIMER);
        vector<vector<string> >* arrangement;
        if (_my_rank == MASTER_RANK) {
            arrangement = params.native_dispatcher ? dispatchLocality(params.input_path.c_str()) : dispatchRan(params.input_path.c_str());
            masterScatter(*arrangement);
            vector<string>& assignedSplits = (*arrangement)[0];
            for (vector<string>::iterator it = assignedSplits.begin(); it != assignedSplits.end(); it++)
                load_graph(it->c_str());
            delete arrangement;
        } else {
            vector<string> assignedSplits;
            slaveScatter(assignedSplits);
            for (vector<string>::iterator it = assignedSplits.begin(); it != assignedSplits.end(); it++)
                load_graph(it->c_str());
        }
        sync_graph();
        cout << "Rank " << _my_rank << " vertex count after sync_graph: " << vertexes.size() << endl;
        message_buffer->init(vertexes);
        init_comm_matrix();
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Load Time", WORKER_TIMER);
    }

    // --- reset_for_query: reactivate all vertices between queries ---
    void reset_for_query()
    {
        active_count = 0;
        MessageBufT* mbuf = (MessageBufT*)get_message_buffer();
        for (auto& container : mbuf->get_v_msg_bufs())
            container.clear();
        for (int i = 0; i < (int)vertexes.size(); i++) {
            vertexes[i]->activate();
            active_count++;
        }
        _vertex_comm_map.clear();
        init_comm_matrix();
        init_machine_matrix();
        _cross_worker_msg_num = 0;
        _cross_machine_msg_num = 0;
        clearBits();
    }

    // --- run_query: compute phase for one source (call after load + reset_for_query) ---
    void run_query(const WorkerParams& params)
    {
        if (_my_rank == MASTER_RANK) {
            if (dirCheck(params.input_path.c_str(), params.output_path.c_str(), _my_rank == MASTER_RANK, params.force_write) == -1)
                exit(-1);
        }
        init_timers();
        ResetTimer(WORKER_TIMER);
        global_step_num = 0;
        long long step_msg_num;
        long long step_vadd_num;
        long long global_msg_num = 0;
        long long global_vadd_num = 0;

        int max_supersteps = 50;
        init_superstep_tracking(max_supersteps);
        double _run_start = MPI_Wtime();

        while (true) {
            global_step_num++;
            ResetTimer(4);
            //===================
            char bits_bor = all_bor(global_bor_bitmap);
            if (getBit(FORCE_TERMINATE_ORBIT, bits_bor) == 1)
                break;
            get_vnum() = all_sum(vertexes.size());
            int wakeAll = getBit(WAKE_ALL_ORBIT, bits_bor);
            if (wakeAll == 0) {
                active_vnum() = all_sum(active_count);
                if (active_vnum() == 0 && getBit(HAS_MSG_ORBIT, bits_bor) == 0)
                    break;
            } else
                active_vnum() = get_vnum();
            //===================
            AggregatorT* agg = (AggregatorT*)get_aggregator();
            if (agg != NULL)
                agg->init();
            //===================
            clearBits();

            int _active_this_step = 0;
            MessageBufT* mbuf = (MessageBufT*)get_message_buffer();
            vector<MessageContainerT>& v_msgbufs = mbuf->get_v_msg_bufs();
            for (int i = 0; i < (int)vertexes.size(); i++) {
                if (vertexes[i]->is_active() || v_msgbufs[i].size() > 0)
                    _active_this_step++;
            }

            double _step_start = MPI_Wtime() - _run_start;
            if (wakeAll == 1)
                all_compute();
            else
                active_compute();
            double _step_end = MPI_Wtime() - _run_start;

            if (global_step_num <= max_supersteps) {
                _worker_step_start[global_step_num][_my_rank] = _step_start;
                _worker_step_end[global_step_num][_my_rank]   = _step_end;
                _worker_step_active[global_step_num][_my_rank] = _active_this_step;
            }

            message_buffer->combine();
            step_msg_num = master_sum_LL(message_buffer->get_total_msg());
            step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
            if (_my_rank == MASTER_RANK) {
                global_msg_num += step_msg_num;
                global_vadd_num += step_vadd_num;
            }
            vector<VertexT*>& to_add = message_buffer->sync_messages();
            agg_sync();
            for (int i = 0; i < to_add.size(); i++)
                add_vertex(to_add[i]);
            to_add.clear();
            //===================
            long long global_cross_worker_msg = all_sum_LL(_cross_worker_msg_num);
            long long global_cross_machine = all_sum_LL(_cross_machine_msg_num);
            _cross_worker_msg_num = 0;
            _cross_machine_msg_num = 0;

            worker_barrier();
            StopTimer(4);
            if (_my_rank == MASTER_RANK) {
                cout << "Superstep " << global_step_num << " done. Time elapsed: " << get_timer(4) << " seconds" << endl;
                cout << "#msgs: " << step_msg_num << ", #vadd: " << step_vadd_num << endl;
                cout << "#cross-worker msgs: " << global_cross_worker_msg << endl;
                cout << "#cross-machine msgs: " << global_cross_machine << endl;
            }
        }
        worker_barrier();
        double _run_end = MPI_Wtime();
        if (_my_rank == MASTER_RANK)
            printf("Query time (src=%d): %.3f seconds\n", params.source_id, _run_end - _run_start);
        StopTimer(WORKER_TIMER);
        PrintTimer("Communication Time", COMMUNICATION_TIMER);
        PrintTimer("- Serialization Time", SERIALIZATION_TIMER);
        PrintTimer("- Transfer Time", TRANSFER_TIMER);
        PrintTimer("Total Computational Time", WORKER_TIMER);
        if (_my_rank == MASTER_RANK)
            cout << "Total #msgs=" << global_msg_num << ", Total #vadd=" << global_vadd_num << endl;

        // Gather all worker timing data to master
        int total_steps = global_step_num;
        for (int s = 1; s <= total_steps; s++) {
            double local_start  = _worker_step_start[s][_my_rank];
            double local_end    = _worker_step_end[s][_my_rank];
            int    local_active = _worker_step_active[s][_my_rank];
            vector<double> all_starts(_num_workers);
            vector<double> all_ends(_num_workers);
            vector<int>    all_actives(_num_workers);
            MPI_Gather(&local_start,  1, MPI_DOUBLE, all_starts.data(),  1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Gather(&local_end,    1, MPI_DOUBLE, all_ends.data(),    1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Gather(&local_active, 1, MPI_INT,    all_actives.data(), 1, MPI_INT,    MASTER_RANK, MPI_COMM_WORLD);
            if (_my_rank == MASTER_RANK) {
                for (int w = 0; w < _num_workers; w++) {
                    _worker_step_start[s][w]  = all_starts[w];
                    _worker_step_end[s][w]    = all_ends[w];
                    _worker_step_active[s][w] = all_actives[w];
                }
            }
        }

        if (_my_rank == MASTER_RANK) {
            char timing_file[256];
            sprintf(timing_file, "worker_timing_src_%d.csv", params.source_id);
            FILE* tf = fopen(timing_file, "w");
            fprintf(tf, "source,superstep,worker,start_time,end_time,duration,active_vertices\n");
            for (int s = 1; s <= total_steps; s++) {
                for (int w = 0; w < _num_workers; w++) {
                    double duration = _worker_step_end[s][w] - _worker_step_start[s][w];
                    fprintf(tf, "%d,%d,%d,%.6f,%.6f,%.6f,%d\n",
                        params.source_id, s, w,
                        _worker_step_start[s][w], _worker_step_end[s][w],
                        duration, _worker_step_active[s][w]);
                }
            }
            fclose(tf);
            char hdfs_mkdir[512];
            sprintf(hdfs_mkdir, "/usr/local/hadoop/bin/hdfs dfs -mkdir -p /comm_traces/src_%d/", params.source_id);
            system(hdfs_mkdir);
            char hdfs_put[512];
            sprintf(hdfs_put, "/usr/local/hadoop/bin/hdfs dfs -put -f %s /comm_traces/src_%d/", timing_file, params.source_id);
            system(hdfs_put);
            remove(timing_file);
        }

        vector<int> my_row(_num_workers);
        for (int i = 0; i < _num_workers; i++)
            my_row[i] = _worker_comm_matrix[_my_rank][i];
        long long total_cross_machine = 0;
        long long total_cross_worker = 0;
        if (_my_rank == MASTER_RANK) {
            for (int w = 1; w < _num_workers; w++) {
                vector<int> row = recv_data<vector<int>>(w);
                for (int i = 0; i < _num_workers; i++)
                    _worker_comm_matrix[w][i] = row[i];
            }
            cout << "\nWorker Communication Matrix (row=src, col=dst):" << endl;
            for (int i = 0; i < _num_workers; i++) {
                for (int j = 0; j < _num_workers; j++)
                    cout << setw(10) << _worker_comm_matrix[i][j];
                cout << endl;
            }
            for (int i = 0; i < _num_workers; i++)
                for (int j = 0; j < _num_workers; j++)
                    if (i != j) total_cross_worker += _worker_comm_matrix[i][j];
            cout << "\nTotal Cross-Worker Messages: " << total_cross_worker << endl;
        } else {
            send_data(my_row, MASTER_RANK);
        }

        int num_machines = (int)_machine_comm_matrix.size();
        vector<int> flat_local(num_machines * num_machines);
        vector<int> flat_global(num_machines * num_machines);
        for (int i = 0; i < num_machines; i++)
            for (int j = 0; j < num_machines; j++)
                flat_local[i * num_machines + j] = _machine_comm_matrix[i][j];
        MPI_Reduce(flat_local.data(), flat_global.data(), num_machines * num_machines,
                   MPI_INT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        if (_my_rank == MASTER_RANK) {
            for (int i = 0; i < num_machines; i++)
                for (int j = 0; j < num_machines; j++)
                    _machine_comm_matrix[i][j] = flat_global[i * num_machines + j];
            cout << "\nMachine Communication Matrix (row=src, col=dst):" << endl;
            for (int i = 0; i < num_machines; i++) {
                for (int j = 0; j < num_machines; j++)
                    cout << setw(10) << _machine_comm_matrix[i][j];
                cout << endl;
            }
            for (int i = 0; i < num_machines; i++)
                for (int j = 0; j < num_machines; j++)
                    if (i != j) total_cross_machine += _machine_comm_matrix[i][j];
            cout << "Total Cross-Machine Messages: " << total_cross_machine << endl;
            cout << "Cross-Machine Ratio: " << (double)total_cross_machine / total_cross_worker << endl;
        }

        int start_node = params.source_id;
        char filename[256];
        sprintf(filename, "vertex_comm_worker_%d_src_%d.csv", _my_rank, start_node);
        FILE* f = fopen(filename, "w");
        if (_my_rank == 0)
            fprintf(f, "source,superstep,src_vertex,dst_vertex,count\n");
        for (auto& [superstep, src_map] : _vertex_comm_map)
            for (auto& [src_vertex, dst_map] : src_map)
                for (auto& [dst_vertex, count] : dst_map)
                    fprintf(f, "%d,%d,%d,%d,%d\n", start_node, superstep, src_vertex, dst_vertex, count);
        fclose(f);

        worker_barrier();
        if (_my_rank == MASTER_RANK) {
            char mkdir_cmd[512];
            sprintf(mkdir_cmd, "/usr/local/hadoop/bin/hdfs dfs -mkdir -p /comm_traces/src_%d/staging/", start_node);
            system(mkdir_cmd);
        }
        worker_barrier();

        char hdfs_cmd[512];
        sprintf(hdfs_cmd, "/usr/local/hadoop/bin/hdfs dfs -put -f %s /comm_traces/src_%d/staging/ 2>/dev/null", filename, start_node);
        system(hdfs_cmd);
        remove(filename);

        if (_my_rank == MASTER_RANK) {
            write_metrics(start_node, global_step_num, global_msg_num,
                get_timer(COMMUNICATION_TIMER), get_timer(SERIALIZATION_TIMER),
                get_timer(TRANSFER_TIMER), get_timer(WORKER_TIMER),
                total_cross_worker, total_cross_machine, _my_rank);
        }

        ResetTimer(WORKER_TIMER);
        dump_partition(params.output_path.c_str());
        StopTimer(WORKER_TIMER);
        PrintTimer("Dump Time", WORKER_TIMER);
    }

    void run(const WorkerParams& params)
    {
        //check path + init
        if (_my_rank == MASTER_RANK) {
            if (dirCheck(params.input_path.c_str(), params.output_path.c_str(), _my_rank == MASTER_RANK, params.force_write) == -1)
                exit(-1);
        }
        init_timers();

        //dispatch splits
        ResetTimer(WORKER_TIMER);
        vector<vector<string> >* arrangement;
        if (_my_rank == MASTER_RANK) {
            arrangement = params.native_dispatcher ? dispatchLocality(params.input_path.c_str()) : dispatchRan(params.input_path.c_str());
            //reportAssignment(arrangement);//DEBUG !!!!!!!!!!
            masterScatter(*arrangement);
            vector<string>& assignedSplits = (*arrangement)[0];
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
            delete arrangement;
        } else {
            vector<string> assignedSplits;
            slaveScatter(assignedSplits);
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
        }

        //send vertices according to hash_id (reduce)
        sync_graph();

        // print vertex distribution 
        cout << "Rank " << _my_rank 
            << " vertex count after sync_graph: " 
            << vertexes.size() << endl;

        message_buffer->init(vertexes);

        init_comm_matrix();
            
        //barrier for data loading
        worker_barrier(); //@@@@@@@@@@@@@
        StopTimer(WORKER_TIMER);
        PrintTimer("Load Time", WORKER_TIMER);

        //=========================================================

        init_timers();
        ResetTimer(WORKER_TIMER);
        //supersteps
        global_step_num = 0;
        long long step_msg_num;
        long long step_vadd_num;
        long long global_msg_num = 0;
        long long global_vadd_num = 0;
        
        // superstep tracking initialization
        int max_supersteps = 50; // safe upper bound
        init_superstep_tracking(max_supersteps);
        double _run_start = MPI_Wtime(); // global reference time

        while (true) {
            global_step_num++;
            ResetTimer(4);
            //===================
            char bits_bor = all_bor(global_bor_bitmap);
            if (getBit(FORCE_TERMINATE_ORBIT, bits_bor) == 1)
                break;
            get_vnum() = all_sum(vertexes.size());
            int wakeAll = getBit(WAKE_ALL_ORBIT, bits_bor);
            if (wakeAll == 0) {
                active_vnum() = all_sum(active_count);
                if (active_vnum() == 0 && getBit(HAS_MSG_ORBIT, bits_bor) == 0)
                    break; //all_halt AND no_msg
            } else
                active_vnum() = get_vnum();
            //===================
            AggregatorT* agg = (AggregatorT*)get_aggregator();
            if (agg != NULL)
                agg->init();
            //===================
            clearBits();

            // Count working vertices BEFORE compute (messages + active)
            int _active_this_step = 0;
            MessageBufT* mbuf = (MessageBufT*)get_message_buffer();
            vector<MessageContainerT>& v_msgbufs = mbuf->get_v_msg_bufs();
            for (int i = 0; i < (int)vertexes.size(); i++) {
                if (vertexes[i]->is_active() || v_msgbufs[i].size() > 0)
                    _active_this_step++;
            }

            // Record start time BEFORE compute
            double _step_start = MPI_Wtime() - _run_start;

            if (wakeAll == 1)
                all_compute();
            else
                active_compute();
            
            // RECORD per-worker end time and active count AFTER compute
            double _step_end = MPI_Wtime() - _run_start;

            // Store locally
            if (global_step_num <= max_supersteps) {
                _worker_step_start[global_step_num][_my_rank] = _step_start;
                _worker_step_end[global_step_num][_my_rank]   = _step_end;
                _worker_step_active[global_step_num][_my_rank] = _active_this_step;
                // printf("DEBUG rank %d step %d active=%d\n", _my_rank, global_step_num, _active_this_step);
            }
            
            message_buffer->combine();
            step_msg_num = master_sum_LL(message_buffer->get_total_msg());
            step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
            if (_my_rank == MASTER_RANK) {
                global_msg_num += step_msg_num;
                global_vadd_num += step_vadd_num;
            }
            vector<VertexT*>& to_add = message_buffer->sync_messages();
            agg_sync();
            for (int i = 0; i < to_add.size(); i++)
                add_vertex(to_add[i]);
            to_add.clear();
            //===================

            long long global_cross_worker_msg = all_sum_LL(_cross_worker_msg_num);
            long long global_cross_machine = all_sum_LL(_cross_machine_msg_num);
            _cross_worker_msg_num = 0;  // reset for next superstep
            _cross_machine_msg_num = 0;

            worker_barrier();
            StopTimer(4);
            if (_my_rank == MASTER_RANK) {
                cout << "Superstep " << global_step_num << " done. Time elapsed: " << get_timer(4) << " seconds" << endl;
                cout << "#msgs: " << step_msg_num << ", #vadd: " << step_vadd_num << endl;
                cout << "#cross-worker msgs: " << global_cross_worker_msg << endl;  // NEW
                cout << "#cross-machine msgs: " << global_cross_machine << endl;
            }
        }
        worker_barrier();
        double _run_end = MPI_Wtime();
        if (_my_rank == MASTER_RANK)
            printf("Query time (src=%d): %.3f seconds\n", params.source_id, _run_end - _run_start);
        StopTimer(WORKER_TIMER);
        PrintTimer("Communication Time", COMMUNICATION_TIMER);
        PrintTimer("- Serialization Time", SERIALIZATION_TIMER);
        PrintTimer("- Transfer Time", TRANSFER_TIMER);
        PrintTimer("Total Computational Time", WORKER_TIMER);
        if (_my_rank == MASTER_RANK)
            cout << "Total #msgs=" << global_msg_num << ", Total #vadd=" << global_vadd_num << endl;

        // Gather all worker timing data to master via MPI_Gather
        int total_steps = global_step_num;

        for (int s = 1; s <= total_steps; s++) {
            // Gather start times from all workers to master
            double local_start  = _worker_step_start[s][_my_rank];
            double local_end    = _worker_step_end[s][_my_rank];
            int    local_active = _worker_step_active[s][_my_rank];

            vector<double> all_starts(_num_workers);
            vector<double> all_ends(_num_workers);
            vector<int>    all_actives(_num_workers);

            MPI_Gather(&local_start,  1, MPI_DOUBLE, all_starts.data(),  1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Gather(&local_end,    1, MPI_DOUBLE, all_ends.data(),    1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Gather(&local_active, 1, MPI_INT,    all_actives.data(), 1, MPI_INT,    MASTER_RANK, MPI_COMM_WORLD);

            if (_my_rank == MASTER_RANK) {
                for (int w = 0; w < _num_workers; w++) {
                    _worker_step_start[s][w]  = all_starts[w];
                    _worker_step_end[s][w]    = all_ends[w];
                    _worker_step_active[s][w] = all_actives[w];
                }
            }
        }

        // Master writes CSV with columns: source, superstep, worker, start_time, end_time, duration, active_vertices
        if (_my_rank == MASTER_RANK) {
            char timing_file[256];
            sprintf(timing_file, "worker_timing_src_%d.csv", params.source_id);
            FILE* tf = fopen(timing_file, "w");
            fprintf(tf, "source,superstep,worker,start_time,end_time,duration,active_vertices\n");
            for (int s = 1; s <= total_steps; s++) {
                for (int w = 0; w < _num_workers; w++) {
                    double duration = _worker_step_end[s][w] - _worker_step_start[s][w];
                    fprintf(tf, "%d,%d,%d,%.6f,%.6f,%.6f,%d\n",
                        params.source_id, s, w,
                        _worker_step_start[s][w],
                        _worker_step_end[s][w],
                        duration,
                        _worker_step_active[s][w]);
                }
            }
            fclose(tf);

            // Upload to HDFS
            char hdfs_mkdir[512];
            sprintf(hdfs_mkdir, "/usr/local/hadoop/bin/hdfs dfs -mkdir -p /comm_traces/src_%d/", params.source_id);
            system(hdfs_mkdir);

            char hdfs_put[512];
            sprintf(hdfs_put, "/usr/local/hadoop/bin/hdfs dfs -put -f %s /comm_traces/src_%d/", timing_file, params.source_id);
            system(hdfs_put);
            remove(timing_file);
        }

        // Every worker sends its row, master collects and prints
        vector<int> my_row(_num_workers);
        for (int i = 0; i < _num_workers; i++)
            my_row[i] = _worker_comm_matrix[_my_rank][i];

        long long total_cross_machine = 0;
        long long total_cross_worker = 0;

        if (_my_rank == MASTER_RANK) {
            for (int w = 1; w < _num_workers; w++) {
                vector<int> row = recv_data<vector<int>>(w);
                for (int i = 0; i < _num_workers; i++)
                    _worker_comm_matrix[w][i] = row[i];
            }
            cout << "\nWorker Communication Matrix (row=src, col=dst):" << endl;
            for (int i = 0; i < _num_workers; i++) {
                for (int j = 0; j < _num_workers; j++)
                    cout << setw(10) << _worker_comm_matrix[i][j];
                cout << endl;
            }

            for (int i = 0; i < _num_workers; i++) {
                for (int j = 0; j < _num_workers; j++) {
                    if (i != j) {
                        total_cross_worker += _worker_comm_matrix[i][j];
                    }
                }
            }

            cout << "\nTotal Cross-Worker Messages: " << total_cross_worker << endl;
        } else {
            send_data(my_row, MASTER_RANK);
        }

        // Get number of machines
        int num_machines = (int)_machine_comm_matrix.size();
        
        // Flatten local matrix for MPI_Reduce
        vector<int> flat_local(num_machines * num_machines);
        vector<int> flat_global(num_machines * num_machines);

        for (int i = 0; i < num_machines; i++) {
            for (int j = 0; j < num_machines; j++) {
                flat_local[i * num_machines + j] = _machine_comm_matrix[i][j];
            }
        }

        // Sum all workers' matrices into master
        MPI_Reduce(
            flat_local.data(), // send buffer
            flat_global.data(), // where the final result goes
            num_machines * num_machines, // num elements
            MPI_INT, // datatype
            MPI_SUM, // operation
            MASTER_RANK, // root
            MPI_COMM_WORLD // communicator
        );

        if (_my_rank == MASTER_RANK) {
            // Reconstruct matrix
            for (int i = 0; i < num_machines; i++) {
                for (int j = 0; j < num_machines; j++) {
                    _machine_comm_matrix[i][j] =
                        flat_global[i * num_machines + j];
                }
            }

            cout << "\nMachine Communication Matrix (row=src, col=dst):" << endl;
            for (int i = 0; i < num_machines; i++) {
                for (int j = 0; j < num_machines; j++)
                    cout << setw(10) << _machine_comm_matrix[i][j];
                cout << endl;
            }

            for (int i = 0; i < num_machines; i++) {
                for (int j = 0; j < num_machines; j++) {
                    if (i != j) {
                        total_cross_machine += _machine_comm_matrix[i][j];
                    }
                }
            }

            cout << "Total Cross-Machine Messages: " << total_cross_machine << endl;
        } 

        // since cross_machine is a subset of cross_worker, we can find out of all inter-worker messages, what fraction requires a network hop?
        // If ratio ≈ 1.0 Almost every cross-worker message goes to another machine.
        // If ratio ≈ 0.25 (for 4 machines) Only 25% of cross-worker traffic crosses machines.
        if (_my_rank == MASTER_RANK) {
            cout << "Cross-Machine Ratio: " << (double)total_cross_machine / total_cross_worker << endl;
        }

        // each worker dumps its own vertex comm entries to a file
        int start_node = params.source_id; // for SSSP specifically

        // each worker dumps its own vertex comm entries to a file, this keeps track of the current superstep now 
        char filename[256];
        sprintf(filename, "vertex_comm_worker_%d_src_%d.csv", _my_rank, start_node);
        FILE* f = fopen(filename, "w");
        if (_my_rank == 0) {
            fprintf(f, "source,superstep,src_vertex,dst_vertex,count\n");
        }
        for (auto& [superstep, src_map] : _vertex_comm_map) {
            for (auto& [src_vertex, dst_map] : src_map) {
                for (auto& [dst_vertex, count] : dst_map) {
                    fprintf(f, "%d,%d,%d,%d,%d\n", 
                        start_node, superstep, src_vertex, dst_vertex, count);
                }
            }
        }
        fclose(f);
        
        // make dir and write to hdfs 
        worker_barrier();
        if (_my_rank == MASTER_RANK) {
            char mkdir_cmd[512];
            sprintf(mkdir_cmd, "/usr/local/hadoop/bin/hdfs dfs -mkdir -p /comm_traces/src_%d/staging/", start_node);
            system(mkdir_cmd);
        }
        worker_barrier();

        char hdfs_cmd[512];
        sprintf(hdfs_cmd, "/usr/local/hadoop/bin/hdfs dfs -put -f %s /comm_traces/src_%d/staging/ 2>/dev/null", filename, start_node);
        system(hdfs_cmd);

        remove(filename);

        if (_my_rank == MASTER_RANK) {
            double comm_time  = get_timer(COMMUNICATION_TIMER);
            double ser_time   = get_timer(SERIALIZATION_TIMER);
            double trans_time = get_timer(TRANSFER_TIMER);
            double compute_time = get_timer(WORKER_TIMER);

            write_metrics(
                start_node,
                global_step_num,
                global_msg_num,
                comm_time,
                ser_time,
                trans_time,
                compute_time,
                total_cross_worker,
                total_cross_machine,
                _my_rank
            );
        }

        // dump graph
        ResetTimer(WORKER_TIMER);
        dump_partition(params.output_path.c_str());
        StopTimer(WORKER_TIMER);
        PrintTimer("Dump Time", WORKER_TIMER);
    }

    //run the worker
    void run(const WorkerParams& params, int num_phases)
    {
        //check path + init
        if (_my_rank == MASTER_RANK) {
            if (dirCheck(params.input_path.c_str(), params.output_path.c_str(), _my_rank == MASTER_RANK, params.force_write) == -1)
                exit(-1);
        }
        init_timers();

        //dispatch splits
        ResetTimer(WORKER_TIMER);
        vector<vector<string> >* arrangement;
        if (_my_rank == MASTER_RANK) {
            arrangement = params.native_dispatcher ? dispatchLocality(params.input_path.c_str()) : dispatchRan(params.input_path.c_str());
            //reportAssignment(arrangement);//DEBUG !!!!!!!!!!
            masterScatter(*arrangement);
            vector<string>& assignedSplits = (*arrangement)[0];
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
            delete arrangement;
        } else {
            vector<string> assignedSplits;
            slaveScatter(assignedSplits);
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
        }

        //send vertices according to hash_id (reduce)
        sync_graph();
        message_buffer->init(vertexes);
        //barrier for data loading
        worker_barrier(); //@@@@@@@@@@@@@
        StopTimer(WORKER_TIMER);
        PrintTimer("Load Time", WORKER_TIMER);

        //=========================================================

        init_timers();
        ResetTimer(WORKER_TIMER);

        for (global_phase_num = 1; global_phase_num <= num_phases; global_phase_num++) {
            if (_my_rank == MASTER_RANK)
                cout << "################ Phase " << global_phase_num << " ################" << endl;

            //supersteps
            global_step_num = 0;
            long long step_msg_num;
            long long step_vadd_num;
            long long global_msg_num = 0;
            long long global_vadd_num = 0;

            while (true) {
                global_step_num++;
                ResetTimer(4);
                //===================
                if (step_num() == 1) {
                    get_vnum() = all_sum(vertexes.size());
                    if (phase_num() > 1)
                        active_vnum() = get_vnum();
                    else
                        active_vnum() = all_sum(active_count);
                    //===================
                    AggregatorT* agg = (AggregatorT*)get_aggregator();
                    if (agg != NULL)
                        agg->init();
                    //===================
                    clearBits();
                    if (phase_num() > 1)
                        all_compute();
                    else
                        active_compute();
                    message_buffer->combine();
                    step_msg_num = master_sum_LL(message_buffer->get_total_msg());
                    step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
                    if (_my_rank == MASTER_RANK) {
                        global_msg_num += step_msg_num;
                        global_vadd_num += step_vadd_num;
                    }
                    vector<VertexT*>& to_add = message_buffer->sync_messages();
                    agg_sync();
                    for (int i = 0; i < to_add.size(); i++)
                        add_vertex(to_add[i]);
                    to_add.clear();
                } else {
                    char bits_bor = all_bor(global_bor_bitmap);
                    if (getBit(FORCE_TERMINATE_ORBIT, bits_bor) == 1)
                        break;
                    get_vnum() = all_sum(vertexes.size());
                    int wakeAll = getBit(WAKE_ALL_ORBIT, bits_bor);
                    if (wakeAll == 0) {
                        active_vnum() = all_sum(active_count);
                        if (active_vnum() == 0 && getBit(HAS_MSG_ORBIT, bits_bor) == 0)
                            break; //all_halt AND no_msg
                    } else
                        active_vnum() = get_vnum();
                    //===================
                    AggregatorT* agg = (AggregatorT*)get_aggregator();
                    if (agg != NULL)
                        agg->init();
                    //===================
                    clearBits();
                    if (wakeAll == 1)
                        all_compute();
                    else if (phase_num() > 1 && step_num() == 1)
                        all_compute();
                    else
                        active_compute();
                    message_buffer->combine();
                    step_msg_num = master_sum_LL(message_buffer->get_total_msg());
                    step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
                    if (_my_rank == MASTER_RANK) {
                        global_msg_num += step_msg_num;
                        global_vadd_num += step_vadd_num;
                    }
                    vector<VertexT*>& to_add = message_buffer->sync_messages();
                    agg_sync();
                    for (int i = 0; i < to_add.size(); i++)
                        add_vertex(to_add[i]);
                    to_add.clear();
                }
                //===================
                worker_barrier();
                StopTimer(4);
                if (_my_rank == MASTER_RANK) {
                    cout << "Superstep " << global_step_num << " done. Time elapsed: " << get_timer(4) << " seconds" << endl;
                    cout << "#msgs: " << step_msg_num << ", #vadd: " << step_vadd_num << endl;
                }
            }
            if (_my_rank == MASTER_RANK) {
                cout << "************ Phase " << global_phase_num << " done. ************" << endl;
                cout << "Total #msgs=" << global_msg_num << ", Total #vadd=" << global_vadd_num << endl;
            }
        }
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Communication Time", COMMUNICATION_TIMER);
        PrintTimer("- Serialization Time", SERIALIZATION_TIMER);
        PrintTimer("- Transfer Time", TRANSFER_TIMER);
        PrintTimer("Total Computational Time", WORKER_TIMER);

        // dump graph
        ResetTimer(WORKER_TIMER);
        dump_partition(params.output_path.c_str());
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Dump Time", WORKER_TIMER);
    }

    // run the worker
    void run(const MultiInputParams& params)
    {
        //check path + init
        if (_my_rank == MASTER_RANK) {
            if (dirCheck(params.input_paths, params.output_path.c_str(), _my_rank == MASTER_RANK, params.force_write) == -1)
                exit(-1);
        }
        init_timers();

        //dispatch splits
        ResetTimer(WORKER_TIMER);
        vector<vector<string> >* arrangement;
        if (_my_rank == MASTER_RANK) {
            arrangement = params.native_dispatcher ? dispatchLocality(params.input_paths) : dispatchRan(params.input_paths);
            //reportAssignment(arrangement);//DEBUG !!!!!!!!!!
            masterScatter(*arrangement);
            vector<string>& assignedSplits = (*arrangement)[0];
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
            delete arrangement;
        } else {
            vector<string> assignedSplits;
            slaveScatter(assignedSplits);
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
        }

        //send vertices according to hash_id (reduce)
        sync_graph();
        message_buffer->init(vertexes);
        //barrier for data loading
        worker_barrier(); //@@@@@@@@@@@@@
        StopTimer(WORKER_TIMER);
        PrintTimer("Load Time", WORKER_TIMER);

        //=========================================================

        init_timers();
        ResetTimer(WORKER_TIMER);
        //supersteps
        global_step_num = 0;
        long long step_msg_num;
        long long step_vadd_num;
        long long global_msg_num = 0;
        long long global_vadd_num = 0;
        while (true) {
            global_step_num++;
            ResetTimer(4);
            //===================
            char bits_bor = all_bor(global_bor_bitmap);
            if (getBit(FORCE_TERMINATE_ORBIT, bits_bor) == 1)
                break;
            get_vnum() = all_sum(vertexes.size());
            int wakeAll = getBit(WAKE_ALL_ORBIT, bits_bor);
            if (wakeAll == 0) {
                active_vnum() = all_sum(active_count);
                if (active_vnum() == 0 && getBit(HAS_MSG_ORBIT, bits_bor) == 0)
                    break; //all_halt AND no_msg
            } else
                active_vnum() = get_vnum();
            //===================
            AggregatorT* agg = (AggregatorT*)get_aggregator();
            if (agg != NULL)
                agg->init();
            //===================
            clearBits();
            if (wakeAll == 1)
                all_compute();
            else
                active_compute();
            message_buffer->combine();
            step_msg_num = master_sum_LL(message_buffer->get_total_msg());
            step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
            if (_my_rank == MASTER_RANK) {
                global_msg_num += step_msg_num;
                global_vadd_num += step_vadd_num;
            }
            vector<VertexT*>& to_add = message_buffer->sync_messages();
            agg_sync();
            for (int i = 0; i < to_add.size(); i++)
                add_vertex(to_add[i]);
            to_add.clear();
            //===================
            worker_barrier();
            StopTimer(4);
            if (_my_rank == MASTER_RANK) {
                cout << "Superstep " << global_step_num << " done. Time elapsed: " << get_timer(4) << " seconds" << endl;
                cout << "#msgs: " << step_msg_num << ", #vadd: " << step_vadd_num << endl;
            }
        }
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Communication Time", COMMUNICATION_TIMER);
        PrintTimer("- Serialization Time", SERIALIZATION_TIMER);
        PrintTimer("- Transfer Time", TRANSFER_TIMER);
        PrintTimer("Total Computational Time", WORKER_TIMER);
        if (_my_rank == MASTER_RANK)
            cout << "Total #msgs=" << global_msg_num << ", Total #vadd=" << global_vadd_num << endl;

        // dump graph
        ResetTimer(WORKER_TIMER);
        dump_partition(params.output_path.c_str());
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Dump Time", WORKER_TIMER);
    }

    //========================== reports machine-level msg# ===============================
    void run_report(const WorkerParams& params, const string reportPath)
    {
        //check path + init
        if (_my_rank == MASTER_RANK) {
            if (dirCheck(params.input_path.c_str(), params.output_path.c_str(), _my_rank == MASTER_RANK, params.force_write) == -1)
                exit(-1);
        }
        init_timers();

        //dispatch splits
        ResetTimer(WORKER_TIMER);
        vector<vector<string> >* arrangement;
        if (_my_rank == MASTER_RANK) {
            arrangement = params.native_dispatcher ? dispatchLocality(params.input_path.c_str()) : dispatchRan(params.input_path.c_str());
            //reportAssignment(arrangement);//DEBUG !!!!!!!!!!
            masterScatter(*arrangement);
            vector<string>& assignedSplits = (*arrangement)[0];
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
            delete arrangement;
        } else {
            vector<string> assignedSplits;
            slaveScatter(assignedSplits);
            //reading assigned splits (map)
            for (vector<string>::iterator it = assignedSplits.begin();
                 it != assignedSplits.end(); it++)
                load_graph(it->c_str());
        }

        //send vertices according to hash_id (reduce)
        sync_graph();
        message_buffer->init(vertexes);
        //barrier for data loading
        worker_barrier(); //@@@@@@@@@@@@@
        StopTimer(WORKER_TIMER);
        PrintTimer("Load Time", WORKER_TIMER);

        //=========================================================
        vector<int> msgNumVec; //$$$$$$$$$$$$$$$$$$$$ added for per-worker msg counting

        init_timers();
        ResetTimer(WORKER_TIMER);
        //supersteps
        global_step_num = 0;
        long long step_msg_num;
        long long step_vadd_num;
        long long global_msg_num = 0;
        long long global_vadd_num = 0;
        while (true) {
            global_step_num++;
            ResetTimer(4);
            //===================
            char bits_bor = all_bor(global_bor_bitmap);
            if (getBit(FORCE_TERMINATE_ORBIT, bits_bor) == 1)
                break;
            get_vnum() = all_sum(vertexes.size());
            int wakeAll = getBit(WAKE_ALL_ORBIT, bits_bor);
            if (wakeAll == 0) {
                active_vnum() = all_sum(active_count);
                if (active_vnum() == 0 && getBit(HAS_MSG_ORBIT, bits_bor) == 0)
                    break; //all_halt AND no_msg
            } else
                active_vnum() = get_vnum();
            //===================
            AggregatorT* agg = (AggregatorT*)get_aggregator();
            if (agg != NULL)
                agg->init();
            //===================
            clearBits();
            if (wakeAll == 1)
                all_compute();
            else
                active_compute();
            message_buffer->combine();
            int my_msg_num = message_buffer->get_total_msg(); //$$$$$$$$$$$$$$$$$$$$ added for per-worker msg counting
            msgNumVec.push_back(my_msg_num); //$$$$$$$$$$$$$$$$$$$$ added for per-worker msg counting
            step_msg_num = master_sum_LL(my_msg_num); //$$$$$$$$$$$$$$$$$$$$ added for per-worker msg counting
            step_vadd_num = master_sum_LL(message_buffer->get_total_vadd());
            if (_my_rank == MASTER_RANK) {
                global_msg_num += step_msg_num;
                global_vadd_num += step_vadd_num;
            }
            vector<VertexT*>& to_add = message_buffer->sync_messages();
            agg_sync();
            for (int i = 0; i < to_add.size(); i++)
                add_vertex(to_add[i]);
            to_add.clear();
            //===================
            worker_barrier();
            StopTimer(4);
            if (_my_rank == MASTER_RANK) {
                cout << "Superstep " << global_step_num << " done. Time elapsed: " << get_timer(4) << " seconds" << endl;
                cout << "#msgs: " << step_msg_num << ", #vadd: " << step_vadd_num << endl;
            }
        }
        worker_barrier();
        StopTimer(WORKER_TIMER);
        PrintTimer("Communication Time", COMMUNICATION_TIMER);
        PrintTimer("- Serialization Time", SERIALIZATION_TIMER);
        PrintTimer("- Transfer Time", TRANSFER_TIMER);
        PrintTimer("Total Computational Time", WORKER_TIMER);
        if (_my_rank == MASTER_RANK)
            cout << "Total #msgs=" << global_msg_num << ", Total #vadd=" << global_vadd_num << endl;

        // dump graph
        ResetTimer(WORKER_TIMER);
        dump_partition(params.output_path.c_str());

        StopTimer(WORKER_TIMER);
        PrintTimer("Dump Time", WORKER_TIMER);

        //dump report
        if (_my_rank != MASTER_RANK) {
            slaveGather(msgNumVec);
        } else {
            vector<vector<int> > report(_num_workers);
            masterGather(report);
            report[MASTER_RANK].swap(msgNumVec);
            //////
            //per line per worker: #msg for step1, #msg for step2, ...
            hdfsFS fs = getHdfsFS();
            hdfsFile out = getWHandle(reportPath.c_str(), fs);
            char buffer[100];
            for (int i = 0; i < _num_workers; i++) {
                for (int j = 0; j < report[i].size(); j++) {
                    sprintf(buffer, "%d ", report[i][j]);
                    hdfsWrite(fs, out, (void*)buffer, strlen(buffer));
                }
                sprintf(buffer, "\n");
                hdfsWrite(fs, out, (void*)buffer, strlen(buffer));
            }
            if (hdfsFlush(fs, out)) {
                fprintf(stderr, "Failed to 'flush' %s\n", reportPath.c_str());
                exit(-1);
            }
            hdfsCloseFile(fs, out);
            hdfsDisconnect(fs);
        }
    }

private:
    HashT hash;
    VertexContainer vertexes;
    int active_count;

    MessageBuffer<VertexT>* message_buffer;
    Combiner<MessageT>* combiner;
    AggregatorT* aggregator;
};

#endif
