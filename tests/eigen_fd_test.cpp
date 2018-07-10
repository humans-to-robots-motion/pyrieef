#include <Eigen/Core>
#include <iostream>
using namespace std;

void print_fd_eigen() {

    double dt =1.;
    uint32_t n = 4; // config space dimension

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, 3 * n);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);

    // Acceleration rows: [I, -2I, I] --> x_{t+1} + x_{t-1} - 2 x_t
    K.block(0, 0, n, n) = I / (dt * dt);
    K.block(0, n, n, n) = -2 * I / (dt * dt);
    K.block(0, 2 * n, n, n) = I / (dt * dt);

    cout << "K : " << endl;
    cout << K << endl;
}

int main() {

    print_fd_eigen();
    return 0;
}
