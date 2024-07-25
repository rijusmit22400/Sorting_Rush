#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>

#define N 1024
#define TOTAL_ELEMENTS (N * N)

// Utility function for checking CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel for generating random numbers
__global__ void setup_rand_kernel(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < TOTAL_ELEMENTS) {
        curand_init(1234, idx, 0, &state[idx]);
    }
}

__global__ void generate_random_numbers(curandState *state, int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < TOTAL_ELEMENTS) {
        data[idx] = curand(&state[idx]) % 1000;  // Random numbers between 0 and 999
    }
}

// 1. Bubble Sort
__global__ void bubbleSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (rowData[j] > rowData[j + 1]) {
                int temp = rowData[j];
                rowData[j] = rowData[j + 1];
                rowData[j + 1] = temp;
            }
        }
    }
}

// 2. Quick Sort
__device__ void quickSortDevice(int *data, int left, int right) {
    if (left >= right) return;
    int pivot = data[(left + right) / 2];
    int i = left, j = right;
    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }
    quickSortDevice(data, left, j);
    quickSortDevice(data, i, right);
}

__global__ void quickSort(int *data) {
    int row = blockIdx.x;
    quickSortDevice(&data[row * N], 0, N - 1);
}

// 3. Merge Sort
__device__ void mergeDevice(int *data, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int *L = new int[n1];
    int *R = new int[n2];
    
    for (i = 0; i < n1; i++)
        L[i] = data[left + i];
    for (j = 0; j < n2; j++)
        R[j] = data[mid + 1 + j];
    
    i = 0;
    j = 0;
    k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            data[k] = L[i];
            i++;
        } else {
            data[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        data[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        data[k] = R[j];
        j++;
        k++;
    }
    
    delete[] L;
    delete[] R;
}

__device__ void mergeSortDevice(int *data, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortDevice(data, left, mid);
        mergeSortDevice(data, mid + 1, right);
        mergeDevice(data, left, mid, right);
    }
}

__global__ void mergeSort(int *data) {
    int row = blockIdx.x;
    mergeSortDevice(&data[row * N], 0, N - 1);
}

// 4. Radix Sort
__device__ void countingSortDevice(int *data, int exp) {
    int output[N];
    int count[10] = {0};
    
    for (int i = 0; i < N; i++)
        count[(data[i] / exp) % 10]++;
    
    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];
    
    for (int i = N - 1; i >= 0; i--) {
        output[count[(data[i] / exp) % 10] - 1] = data[i];
        count[(data[i] / exp) % 10]--;
    }
    
    for (int i = 0; i < N; i++)
        data[i] = output[i];
}

__global__ void radixSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    int max = rowData[0];
    for (int i = 1; i < N; i++)
        if (rowData[i] > max)
            max = rowData[i];
    
    for (int exp = 1; max / exp > 0; exp *= 10)
        countingSortDevice(rowData, exp);
}

// 5. Insertion Sort
__global__ void insertionSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    for (int i = 1; i < N; i++) {
        int key = rowData[i];
        int j = i - 1;
        
        while (j >= 0 && rowData[j] > key) {
            rowData[j + 1] = rowData[j];
            j = j - 1;
        }
        rowData[j + 1] = key;
    }
}

// 6. Selection Sort
__global__ void selectionSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    for (int i = 0; i < N - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < N; j++)
            if (rowData[j] < rowData[min_idx])
                min_idx = j;
        
        int temp = rowData[min_idx];
        rowData[min_idx] = rowData[i];
        rowData[i] = temp;
    }
}

// 7. Heap Sort
__device__ void heapifyDevice(int *data, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && data[left] > data[largest])
        largest = left;
    
    if (right < n && data[right] > data[largest])
        largest = right;
    
    if (largest != i) {
        int temp = data[i];
        data[i] = data[largest];
        data[largest] = temp;
        
        heapifyDevice(data, n, largest);
    }
}

__global__ void heapSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    for (int i = N / 2 - 1; i >= 0; i--)
        heapifyDevice(rowData, N, i);
    
    for (int i = N - 1; i > 0; i--) {
        int temp = rowData[0];
        rowData[0] = rowData[i];
        rowData[i] = temp;
        
        heapifyDevice(rowData, i, 0);
    }
}

// 8. Cocktail Sort
__global__ void cocktailSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    bool swapped = true;
    int start = 0;
    int end = N - 1;
    
    while (swapped) {
        swapped = false;
        
        for (int i = start; i < end; i++) {
            if (rowData[i] > rowData[i + 1]) {
                int temp = rowData[i];
                rowData[i] = rowData[i + 1];
                rowData[i + 1] = temp;
                swapped = true;
            }
        }
        
        if (!swapped)
            break;
        
        swapped = false;
        end--;
        
        for (int i = end - 1; i >= start; i--) {
            if (rowData[i] > rowData[i + 1]) {
                int temp = rowData[i];
                rowData[i] = rowData[i + 1];
                rowData[i + 1] = temp;
                swapped = true;
            }
        }
        
        start++;
    }
}

// 9. Shell Sort
__global__ void shellSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    for (int gap = N / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < N; i++) {
            int temp = rowData[i];
            int j;
            for (j = i; j >= gap && rowData[j - gap] > temp; j -= gap)
                rowData[j] = rowData[j - gap];
            rowData[j] = temp;
        }
    }
}

// 10. Comb Sort
__global__ void combSort(int *data) {
    int row = blockIdx.x;
    int *rowData = &data[row * N];
    
    int gap = N;
    bool swapped = true;
    
    while (gap != 1 || swapped) {
        gap = (gap * 10) / 13;
        if (gap < 1)
            gap = 1;
        
        swapped = false;
        
        for (int i = 0; i < N - gap; i++) {
            if (rowData[i] > rowData[i + gap]) {
                int temp = rowData[i];
                rowData[i] = rowData[i + gap];
                rowData[i + gap] = temp;
                swapped = true;
            }
        }
    }
}

// Kernel to calculate row sums
__global__ void calculateRowSums(int *data, int *rowSums) {
    int row = blockIdx.x;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += data[row * N + i];
    }
    rowSums[row] = sum;
}

// Function to sort rows based on their sums (on CPU for simplicity)
void sortRowsBySums(int *data, int *rowSums) {
    std::vector<std::pair<int, int>> rowSumPairs(N);
    for (int i = 0; i < N; i++) {
        rowSumPairs[i] = {rowSums[i], i};
    }
    
    std::sort(rowSumPairs.begin(), rowSumPairs.end());
    
    int *tempData = new int[TOTAL_ELEMENTS];
    memcpy(tempData, data, TOTAL_ELEMENTS * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        int sourceRow = rowSumPairs[i].second;
        memcpy(&data[i * N], &tempData[sourceRow * N], N * sizeof(int));
    }
    
    delete[] tempData;
}

// Main function

int main() {
    int *d_data, *d_rowSums;
    curandState *d_state;
    
    cudaMalloc(&d_data, TOTAL_ELEMENTS * sizeof(int));
    cudaMalloc(&d_rowSums, N * sizeof(int));
    cudaMalloc(&d_state, TOTAL_ELEMENTS * sizeof(curandState));
    
    // Generate random numbers
    setup_rand_kernel<<<(TOTAL_ELEMENTS + 255) / 256, 256>>>(d_state);
    generate_random_numbers<<<(TOTAL_ELEMENTS + 255) / 256, 256>>>(d_state, d_data);
    cudaCheckError();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<std::pair<float, const char*>> timings;
    
    // Benchmark all sorting algorithms
    const int numAlgorithms = 10;
    void (*sortingKernels[numAlgorithms])(int*) = {
        bubbleSort, quickSort, mergeSort, radixSort, insertionSort,
        selectionSort, heapSort, cocktailSort, shellSort, combSort
    };
    const char* algorithmNames[numAlgorithms] = {
        "Bubble Sort", "Quick Sort", "Merge Sort", "Radix Sort", "Insertion Sort",
        "Selection Sort", "Heap Sort", "Cocktail Sort", "Shell Sort", "Comb Sort"
    };
    
    for (int i = 0; i < numAlgorithms; i++) {
        printf("%s begins...\n", algorithmNames[i]);
        cudaEventRecord(start);
        sortingKernels[i]<<<N, 1>>>(d_data);
        cudaCheckError();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float sortTime;
        cudaEventElapsedTime(&sortTime, start, stop);
        printf("%s completed in %.3f ms\n", algorithmNames[i], sortTime);
        timings.push_back({sortTime, algorithmNames[i]});
        
        // Reset data for next algorithm
        generate_random_numbers<<<(TOTAL_ELEMENTS + 255) / 256, 256>>>(d_state, d_data);
        cudaCheckError();
    }
    
    // Calculate row sums
    calculateRowSums<<<N, 1>>>(d_data, d_rowSums);
    cudaCheckError();
    
    // Copy data and row sums back to host
    int *h_data = new int[TOTAL_ELEMENTS];
    int *h_rowSums = new int[N];
    cudaMemcpy(h_data, d_data, TOTAL_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowSums, d_rowSums, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Sort rows by sums
    sortRowsBySums(h_data, h_rowSums);
    
    // Copy sorted data back to device
    cudaMemcpy(d_data, h_data, TOTAL_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Total number of elements sorted: %d\n", TOTAL_ELEMENTS);
    
    // Ranking
    std::sort(timings.begin(), timings.end());
    
    printf("\nRanking of sorting algorithms:\n");
    for (int i = 0; i < timings.size(); i++) {
        printf("%d. %s: %.3f ms\n", i + 1, timings[i].second, timings[i].first);
    }
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_rowSums);
    cudaFree(d_state);
    delete[] h_data;
    delete[] h_rowSums;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}