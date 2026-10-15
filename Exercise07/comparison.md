# Exercise 7: Summary Comparison & Analysis

## 1. Collective Communication Comparison Table

| Program | Distribution Collective | Collection / Reduction Collective | Memory Allocation per Process | Root Manual Summation Loop? | Where Final Result is Available |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exercise 1 (`sum_bcast.c`)** | `MPI_Bcast` | Point-to-point (`MPI_Send` / `MPI_Recv`) | **Full array** ($N$ ints = 4 MB) on **every** process | **Yes** (root receives and accumulates in an $O(P)$ loop) | **Root only** (Rank 0) |
| **Exercise 2 (`sum_scatter.c`)**| `MPI_Scatter` | Point-to-point (`MPI_Send` / `MPI_Recv`) | **Root:** Full array ($N$ ints)<br>**Non-root:** Local chunk only ($N/P$ ints) | **Yes** (root receives and accumulates in an $O(P)$ loop) | **Root only** (Rank 0) |
| **Exercise 3 (`sum_gather.c`)** | `MPI_Scatter` | `MPI_Gather` | **Root:** Full array + `all_sums[P]`<br>**Non-root:** Local chunk only ($N/P$ ints) | **Yes** (root loops over gathered `all_sums[0..P-1]`) | **Root only** (Rank 0) |
| **Exercise 4 (`sum_reduce.c`)** | `MPI_Scatter` | `MPI_Reduce` (`MPI_SUM`) | **Root:** Full array ($N$ ints)<br>**Non-root:** Local chunk only ($N/P$ ints) | **No** (MPI automatically combines values in $O(\log P)$ steps) | **Root only** (Rank 0) |
| **Exercise 5 (`sum_allreduce.c`)**| `MPI_Scatter` | `MPI_Allreduce` (`MPI_SUM`) | **Root:** Full array ($N$ ints)<br>**Non-root:** Local chunk only ($N/P$ ints) | **No** (MPI automatically combines and broadcasts in one step) | **All processes** (Every rank has identical global total) |
| **Exercise 6 (`sum_scan.c`)** | `MPI_Scatter` | `MPI_Scan` (`MPI_SUM`) | **Root:** Full array ($N$ ints)<br>**Non-root:** Local chunk only ($N/P$ ints) | **No** (MPI computes prefix reduction automatically) | **Different per rank** (Rank $r$ holds cumulative sum from rank $0$ to $r$; last rank holds global total) |

---

## 2. Performance & Execution Time Benchmark

The table below presents the recorded execution times (in seconds) for summing $N = 1,000,000$ integers across 2, 4, and 8 processes on the local testbed (AMD64 Open MPI 4.1.6):

| Program | Time (2 processes) | Time (4 processes) | Time (8 processes) | Scalability Trend |
| :--- | :---: | :---: | :---: | :--- |
| **`sum_bcast`** | 0.0046 s | 0.0094 s | 0.0192 s | **Degrades significantly** with increasing $P$ |
| **`sum_scatter`** | 0.0061 s | 0.0024 s | 0.0024 s | Good speedup from 2 to 4 processes |
| **`sum_gather`** | 0.0070 s | 0.0034 s | 0.0021 s | Good scaling; single collective replaces point-to-point loop |
| **`sum_reduce`** | 0.0074 s | 0.0027 s | 0.0023 s | **Fastest & most scalable reduction to root** |
| **`sum_allreduce`** | 0.0056 s | 0.0049 s | 0.0059 s | Consistent low latency; distributes result to all processes |
| **`sum_scan`** | 0.0092 s | 0.0108 s | 0.0162 s | Higher communication dependency due to prefix pipeline |

### Explanation of Performance Differences:
1. **`sum_bcast` Memory and Communication Bottleneck**:
   - In `sum_bcast`, root sends the entire 1,000,000-integer buffer (4 MB) to *every* process. As $P$ increases to 8, root transmits $7 \times 4\text{ MB} = 28\text{ MB}$ of data over the network/shared memory.
   - Furthermore, the point-to-point `MPI_Send`/`MPI_Recv` collection on root requires $P-1$ sequential round-trips ($O(P)$ serialized execution).
2. **`sum_scatter` Efficiency**:
   - `MPI_Scatter` splits the data so root transmits only $4\text{ MB}$ in total across all processes (each process receives only $1/P$-th of the array, e.g., 125,000 integers for $P=8$). This drastically reduces memory footprint and cache thrashing.
3. **`sum_reduce` vs `sum_gather` vs Point-to-Point**:
   - `MPI_Reduce` is faster and more scalable than `MPI_Gather` and `MPI_Send`/`MPI_Recv` because it employs an internal **binomial tree** or **recursive halving/doubling** algorithm running in $O(\log P)$ steps.
   - It computes the partial sums during the communication tree itself, eliminating the need for root to receive $P$ separate numbers and execute a serial accumulation loop.
4. **`sum_allreduce` vs `sum_reduce`**:
   - `MPI_Allreduce` has slightly more overhead than `MPI_Reduce` because all ranks must receive the final sum (often implemented via a butterfly algorithm or reduce-scatter followed by allgather), but it avoids a separate `MPI_Bcast` call when every rank needs the global total.
5. **`sum_scan` Overhead**:
   - `MPI_Scan` requires a prefix communication pattern across ordered ranks, meaning each rank's value depends on preceding ranks. While it enables each rank to obtain its cumulative sum, it has higher inter-rank communication synchronization compared to symmetric tree reductions.

---

## 3. Thinking Question: Choosing `MPI_Scan` over `MPI_Allreduce`

### When would you choose `MPI_Scan` over `MPI_Allreduce`?
You would choose `MPI_Scan` when each process requires **local awareness of its relative position or cumulative running total within the global dataset**, rather than just knowing a single overall total.

### Concrete Example: Parallel Stream Compaction & Global Output File Offsets
Suppose $P$ parallel processes filter a large distributed dataset (e.g., finding valid database records or edge vertices in a graph). Each process $i$ finds a variable number of valid items, denoted as $C_i$:
- Rank 0 finds $C_0 = 150$ items.
- Rank 1 finds $C_1 = 300$ items.
- Rank 2 finds $C_2 = 200$ items.
- Rank 3 finds $C_3 = 250$ items.

All processes must write their filtered records directly into a single shared file or contiguous memory buffer without overlapping and without sending all the records to root:
- With `MPI_Allreduce`, every process learns that there are $150 + 300 + 200 + 250 = 900$ items total. However, **no process knows where in the file its chunk starts!**
- With `MPI_Scan`, each process $i$ performs an inclusive scan on $C_i$:
  - Rank 0 gets `prefix_sum = 150` $\rightarrow$ writes items to offset $[0, 150)$.
  - Rank 1 gets `prefix_sum = 450` $\rightarrow$ `offset = prefix_sum - C_1 = 150` $\rightarrow$ writes items to offset $[150, 450)$.
  - Rank 2 gets `prefix_sum = 650` $\rightarrow$ `offset = prefix_sum - C_2 = 450` $\rightarrow$ writes items to offset $[450, 650)$.
  - Rank 3 gets `prefix_sum = 900` $\rightarrow$ `offset = prefix_sum - C_3 = 650` $\rightarrow$ writes items to offset $[650, 900)$.

Using `MPI_Scan`, every process calculates its exact, conflict-free global write offset in $O(\log P)$ time with zero centralized coordination!
