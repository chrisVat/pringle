#include "pregel_app_pagerank.h"

int main(int argc, char* argv[]){
	init_workers();

	cout << "Rank: " << _my_rank << " | Size: " << _num_workers << endl;
	
	pregel_pagerank("/smallTwitchFolder", "/outputSmallTwitchFolder", true);
	worker_finalize();
	return 0;
}
