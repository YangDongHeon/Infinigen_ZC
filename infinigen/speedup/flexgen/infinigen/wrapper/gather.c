// gather_workers.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

// Structure holding the gather task parameters.
typedef struct {
    const float* src;
    float* dst;
    const int* sel;
    int num_sel;
} GatherTask;

// Global shared task and synchronization objects.
pthread_mutex_t taskMutex   = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  taskCond    = PTHREAD_COND_INITIALIZER;
bool taskAvailable = false;
bool shutdownFlag  = false;
GatherTask currentTask;

// Barrier for task completion.
pthread_mutex_t barrierMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  barrierCond  = PTHREAD_COND_INITIALIZER;
int barrierCount = 0;

// Thread pool globals.
int numThreads = 0;
pthread_t* workerThreads = NULL;
int* threadIds = NULL;  // Each thread's identifier

int taskID = 0;

// Worker thread function.
void* worker_function(void* arg) {
    int threadId = *((int*) arg);
    int lastTaskID = -1; // Initially, no task processed.
    while (1) {
        // Wait for a task or shutdown signal.
        pthread_mutex_lock(&taskMutex);
        while (!taskAvailable && !shutdownFlag) {
            pthread_cond_wait(&taskCond, &taskMutex);
        }
        if (shutdownFlag) {
            pthread_mutex_unlock(&taskMutex);
            break;
        }
        // Only process if a new task is available.
        if (taskID <= lastTaskID) {
            // Already processed this task; just wait.
            pthread_mutex_unlock(&taskMutex);
            continue;
        }

        // Copy the current task locally.
        lastTaskID = taskID;
        GatherTask localTask = currentTask;
        pthread_mutex_unlock(&taskMutex);

        // Determine work range.
        int total = localTask.num_sel;
        int chunk_size = total / numThreads;
        int remainder = total % numThreads;
        int start, count;
        if (threadId < remainder) {
            count = chunk_size + 1;
            start = threadId * count;
        } else {
            count = chunk_size;
            start = threadId * count + remainder;
        }
        int end = start + count;

        // Perform the gather: copy selected elements from src to dst.
        for (int i = start; i < end; i++) {
            localTask.dst[i] = localTask.src[ localTask.sel[i] ];
        }

        // Barrier: signal that this thread has finished its work.
        pthread_mutex_lock(&barrierMutex);
        barrierCount++;
        if (barrierCount == numThreads) {
            pthread_cond_signal(&barrierCond);
        }
        pthread_mutex_unlock(&barrierMutex);
    }
    return NULL;
}

// Initialize the persistent worker thread pool.
// Call this from your Python initialization (via ctypes).
void init_gather_workers(int threads) {
    numThreads = threads;
    shutdownFlag = false;
    taskAvailable = false;
    workerThreads = (pthread_t*) malloc(sizeof(pthread_t) * numThreads);
    threadIds = (int*) malloc(sizeof(int) * numThreads);
    for (int i = 0; i < numThreads; i++) {
        threadIds[i] = i;
        pthread_create(&workerThreads[i], NULL, worker_function, &threadIds[i]);
    }
}

// Shutdown the worker threads.
// Call this during cleanup from Python.
void shutdown_gather_workers() {
    pthread_mutex_lock(&taskMutex);
    shutdownFlag = true;
    pthread_cond_broadcast(&taskCond);
    pthread_mutex_unlock(&taskMutex);
    for (int i = 0; i < numThreads; i++) {
        pthread_join(workerThreads[i], NULL);
    }
    free(workerThreads);
    free(threadIds);
}

// Start a new gather task.
// This function sets up the task and signals the worker threads,
// then returns immediately so the main thread can perform other work.
void start_gather_task(const float* src, float* dst, const int* sel, int num_sel) {
    pthread_mutex_lock(&taskMutex);
    currentTask.src = src;
    currentTask.dst = dst;
    currentTask.sel = sel;
    currentTask.num_sel = num_sel;
    taskAvailable = true;
    taskID++;  // Increment the task generation counter.
    pthread_cond_broadcast(&taskCond);
    pthread_mutex_unlock(&taskMutex);
}

// Wait for the current gather task to complete.
// This function blocks until all worker threads have finished processing.
void wait_for_gather_task() {
    pthread_mutex_lock(&barrierMutex);
    while (barrierCount < numThreads) {
        pthread_cond_wait(&barrierCond, &barrierMutex);
    }
    // Reset barrier count for the next task.
    barrierCount = 0;
    pthread_mutex_unlock(&barrierMutex);

    // Optionally, clear the task flag.
    pthread_mutex_lock(&taskMutex);
    taskAvailable = false;
    pthread_mutex_unlock(&taskMutex);
}

// If desired, you can provide a main function for testing this library standalone.
// For testing as a standalone executable.
#ifdef TEST_MAIN
int main() {
    // Initialize worker threads (e.g., 4 threads).
    init_gather_workers(4);

    // Create test data.
    int num = 10000;
    int num_sel = 500;
    float* src = (float*) malloc(sizeof(float) * num);
    float* dst = (float*) malloc(sizeof(float) * num_sel);
    int* sel = (int*) malloc(sizeof(int) * num_sel);
    for (int i = 0; i < num; i++) {
        src[i] = (float) i;
    }
    for (int i = 0; i < num_sel; i++) {
        sel[i] = i;  // Each index is valid.
    }

    // Start a gather task.
    start_gather_task(src, dst, sel, num_sel);

    // Main thread can do other work here.
    printf("Main thread is doing other work...\n");

    // Later, wait for the gather task to complete.
    wait_for_gather_task();

    // (Optional) Verify the results.
    for (int i = 0; i < num_sel; i++) {
        if (dst[i] != src[ sel[i] ]) {
            printf("Error at index %d: dst=%f, expected=%f\n", i, dst[i], src[ sel[i] ]);
        }
    }
    printf("Gather task completed.\n");

    // Clean up.
    free(src);
    free(dst);
    free(sel);
    shutdown_gather_workers();
    return 0;
}
#endif
