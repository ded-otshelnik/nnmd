#include "head.hpp"

std::vector<std::tuple<double, double, double>> params_for_G2(int n_radial, double r_cutoff) {
    std::vector<std::tuple<double, double, double>> params;
    for (int i = 0; i < n_radial; ++i) {
        double rs = r_cutoff / (n_radial - 1);
        double eta = 5 * std::log(10) / (4 * std::pow(rs, 2));
        params.emplace_back(r_cutoff, rs * i, eta);
    }
    return params;
}

std::vector<std::tuple<double, double, int, double>> params_for_G4(int n_angular, double r_cutoff) {
    std::vector<std::tuple<double, double, int, double>> params;
    int ind = 1;
    for (int i = 0; i < n_angular; ++i) {
        double eta = 2 * std::log(10) / std::pow(r_cutoff, 2);
        double xi = 1 + i * 30 / (n_angular - 2);
        for (int lambd : {1, -1}) {
            params.emplace_back(r_cutoff, eta, lambd, xi);
            if (ind >= n_angular) {
                break;
            }
            ind++;
        }
    }
    return params;
}

torch::Tensor f_cutoff(const torch::Tensor& r, double cutoff) {
    return torch::where(torch::abs(r - cutoff) >= 1e-8, 0.5 * (torch::cos(r * M_PI / cutoff) + 1), torch::zeros_like(r));
}

torch::Tensor g2_function(const torch::Tensor& r, double cutoff, double eta, double rs) {
    auto fc = f_cutoff(r, cutoff);
    return torch::exp(-eta * torch::pow(r - rs, 2)) * fc;
}

torch::Tensor g4_function(const torch::Tensor& rij, const torch::Tensor& rjk, const torch::Tensor& rik, const torch::Tensor& cos_theta, double cutoff, double eta, double zeta, int lambd) {
    auto fc_ij = f_cutoff(rij, cutoff);
    auto fc_ik = f_cutoff(rik, cutoff);
    auto fc_jk = f_cutoff(rjk, cutoff);
    auto fc = fc_ij * fc_ik * fc_jk;

    auto term1 = torch::pow(1 + lambd * cos_theta, zeta);
    auto term2 = torch::exp(-eta * (torch::pow(rij, 2) + torch::pow(rik, 2) + torch::pow(rjk, 2)));
    return std::pow(2, 1 - zeta) * term1 * term2 * fc;
}

torch::Tensor calculate_distances(const torch::Tensor& x) {
    /*
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    */
    auto x_norm = (x.pow(2).sum(1)).view({-1, 1});
    torch::Tensor y_t, y_norm;

    y_t = x.transpose(0, 1);
    y_norm = x_norm.view({1, -1});
    

    auto dist = x_norm + y_norm - 2.0 * torch::mm(x, y_t);
    return torch::clamp(dist, 0.0, std::numeric_limits<double>::infinity());
}

torch::Tensor calculate_cosines(const torch::Tensor& rij, const torch::Tensor& rik, const torch::Tensor& rjk) {
    return torch::where(torch::abs(2 * rij * rik) <= 10e-4,
                        (torch::pow(rij, 2) + torch::pow(rik, 2) - torch::pow(rjk, 2)) / (2 * rij * rik),
                        torch::zeros_like(rij));
}

torch::Tensor calculate_mask(const torch::Tensor& cartesians) {
    int n_atoms = cartesians.size(0);
    auto mask = (torch::arange(n_atoms, cartesians.device()).unsqueeze(1) != torch::arange(n_atoms, cartesians.device())).unsqueeze(2);
    mask = mask & mask.transpose(0, 1) & mask.transpose(1, 2);
    return mask;
}

std::tuple<torch::Tensor, torch::Tensor>  _internal(const torch::Tensor& cart,
    const vector<int> features, const vector<vector<double>> params) {
        torch::Tensor distances = calculate_distances(cart);
        torch::Tensor mask = calculate_mask(cart);
        torch::Tensor rij = distances.unsqueeze(2);
        torch::Tensor rik = distances.unsqueeze(1);
        torch::Tensor rjk = distances.unsqueeze(0);
        torch::Tensor cosines = calculate_cosines(rij, rik, rjk);

        torch::Tensor g_values;
        
        std::vector<torch::Tensor> g_struct;
        std::vector<torch::Tensor> dg_struct;

        std::vector<torch::Tensor> g_values_vec;

        for (size_t g_func = 0; g_func < features.size(); g_func++) {
            g_values_vec.clear();
            if (features[g_func] == 2) {
                for (int i = 0; i < cart.size(0); ++i) {
                    g_values_vec.push_back(g2_function(distances[i],
                     params[g_func][0], params[g_func][1], params[g_func][2]).sum());
                }
                g_values = torch::stack(g_values_vec);
            } else if (features[g_func] == 4) {
                std::vector<torch::Tensor> g_values_vec;
                for (int i = 0; i < cart.size(0); ++i) {
                    torch::Tensor g_value = g4_function(rij, rjk, rik, cosines[i],
                     params[g_func][0], params[g_func][1], params[g_func][2], params[g_func][3]);

                    g_value = g_value * mask.to(torch::kFloat);

                    g_value = torch::nan_to_num(g_value, 0.0);

                    g_values_vec.push_back(g_value.sum());
                }
                g_values = torch::stack(g_values_vec);
            }

            g_values = torch::clamp(g_values, 0.0, 1.0);
            g_struct.push_back(g_values);
            dg_struct.push_back(torch::autograd::grad({g_values.sum()}, {cart}, {}, true, true)[0]);
        }
        return std::make_tuple(torch::stack(g_struct, -1), torch::stack(dg_struct, -1).permute({0, 2, 1}));
};

std::tuple<torch::Tensor, torch::Tensor> calculate_input(const torch::Tensor& cartesians,
    const vector<int> features, const vector<vector<double>> params) {
    py::gil_scoped_release release;
    std::vector<torch::Tensor> g_list, dG_list;
    for (int i = 0; i < cartesians.size(0); ++i) {
        cout << "Calculating input for atom " << i << endl;
        auto [g, dG] = _internal(cartesians[i], features, params);
        g_list.push_back(g);
        dG_list.push_back(dG);
    }
    auto g = torch::stack(g_list);
    auto dG = torch::stack(dG_list);

    return std::make_tuple(g, dG);
}