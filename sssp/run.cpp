#include "pregel_app_sssp.h"

int main(int argc, char* argv[]){
	if (argc < 4){
        printf("Usage: %s <source_id> <input> <output>\n", argv[0]);
        return -1;
    }

	int source = atoi(argv[1]);
    string input = argv[2];
    string output = argv[3];
	
	init_workers();
	pregel_sssp(source, input, output, true);
	worker_finalize();
	return 0;
}
