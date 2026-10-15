CC = mpicc
CFLAGS = -Wall -O2
NP = 4

TARGETS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(TARGETS)

sum_bcast: sum_bcast.c
	$(CC) $(CFLAGS) -o sum_bcast sum_bcast.c

sum_scatter: sum_scatter.c
	$(CC) $(CFLAGS) -o sum_scatter sum_scatter.c

sum_gather: sum_gather.c
	$(CC) $(CFLAGS) -o sum_gather sum_gather.c

sum_reduce: sum_reduce.c
	$(CC) $(CFLAGS) -o sum_reduce sum_reduce.c

sum_allreduce: sum_allreduce.c
	$(CC) $(CFLAGS) -o sum_allreduce sum_allreduce.c

sum_scan: sum_scan.c
	$(CC) $(CFLAGS) -o sum_scan sum_scan.c

run: all
	@echo "=========================================="
	@echo "Running sum_bcast with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_bcast
	@echo ""
	@echo "=========================================="
	@echo "Running sum_scatter with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_scatter
	@echo ""
	@echo "=========================================="
	@echo "Running sum_gather with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_gather
	@echo ""
	@echo "=========================================="
	@echo "Running sum_reduce with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_reduce
	@echo ""
	@echo "=========================================="
	@echo "Running sum_allreduce with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_allreduce
	@echo ""
	@echo "=========================================="
	@echo "Running sum_scan with $(NP) processes"
	@echo "=========================================="
	mpirun --oversubscribe -np $(NP) ./sum_scan
	@echo ""

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
