#include <iostream>
#include "lightmamba.h"

// Function to print a 1D array
void print_array(const char* name, DTYPE arr[N]) {
    std::cout << name << ": [";
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i];
        if (i < N - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void print_2d_array(const char* name, DTYPE arr[M][N])
{
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "  Row " << i << ": [";
        for (int j = 0; j < N; ++j) {
            std::cout << arr[i][j];
            if (j < N - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    DTYPE kernel[K];
    DTYPE A[N], B[N], C[N], D[N];
    DTYPE X[INPUT_DIM], Z[N];
    DTYPE H0[M][N], H1[M][N];
    DTYPE bias[N], delta[N];
    DTYPE out[N];

    for (int i = 0; i < K; ++i) {
        kernel[i] = DTYPE(0.1);
    }
    for (int i = 0; i < N; ++i) {
        A[i] = DTYPE(0.5);
        B[i] = DTYPE(0.2);
        C[i] = DTYPE(0.8);
        D[i] = DTYPE(0.3);
        Z[i] = DTYPE(0.9);
        delta[i] = DTYPE(0.1);
        bias[i] = DTYPE(0.05);
        out[i] = 0;
    }
    for (int i = 0; i < INPUT_DIM; ++i) {
            X[i] = DTYPE(1.0);
        }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            H0[i][j] = DTYPE(0.1);
        }
    }

    SSMU(kernel, A, B, C, D, X, Z, H0, H1, delta, bias, out);

    // Print the final output for verification
    print_array("Final Output", out);
    print_2d_array("Updated State H1", H1);

    // In a full test bench, you would compare 'out' with a known good
    // reference output and report success or failure.
    // For example:
    // DTYPE expected_out[N] = { ... };
    // for (int i = 0; i < N; ++i) {
    //     if (out[i] != expected_out[i]) {
    //         std::cout << "Test failed at index " << i << std::endl;
    //         return 1;
    //     }
    // }
    // std::cout << "Test passed!" << std::endl;

    return 0;
}
