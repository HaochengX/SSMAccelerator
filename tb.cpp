#include <iostream>
#include "ssm_accelerator.h"

int main() {
    DTYPE A[M][N], B[M][N], C[M][N], D[M][N], X[M][N], H0[M][N];
    DTYPE H1[M][N], Y[M][N];
    DTYPE d_scalar, b_scalar;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * N + j + 1;
            B[i][j] = (i * N + j + 1) * 0.5;
            C[i][j] = (i * N + j + 1) * 0.2;
            D[i][j] = (i * N + j + 1) * 0.1;
            X[i][j] = (i * N + j + 1) * 0.7;
            H0[i][j] = (i * N + j + 1) * 0.3;
        }
    }
    d_scalar = 0.5;
    b_scalar = 0.2;

    std::cout << "Calling ssm function..." << std::endl;
    ssm(A, B, C, D, X, H0, H1, Y, d_scalar, b_scalar);
    std::cout << "ssm function call complete." << std::endl;

    std::cout << "\nH1 Output Matrix:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << H1[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nY Output Matrix:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << Y[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    // DTYPE expected_H1[M][N];
    // DTYPE expected_Y[M][N];
    // bool H1_match = true;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (std::abs(H1[i][j].to_double() - expected_H1[i][j].to_double()) > 1e-6) {
    //             H1_match = false;
    //             break;
    //         }
    //     }
    //     if (!H1_match) break;
    // }
    // if (H1_match) {
    //     std::cout << "\nH1 matches expected output!" << std::endl;
    // } else {
    //     std::cout << "\nH1 mismatch detected!" << std::endl;
    // }

    return 0;
}
