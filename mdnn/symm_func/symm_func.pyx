# cython: language_level=3
# distutils: language = c++

cdef extern from "symmetric_functions.h":
    double G1(const double rij, const double rc, double* out_dg) except +
    double G2(const double rij, const double rc, const double eta, const double rs, double* out_dg) except +
    double G3(const double rij, const double rc, const double k, double* out_dg) except +
    double G4(const double rij, const double rik, const double rjk, const double rc, const double eta, const double _lambda, const double xi, const double cos_v, const double* dcos_v, double* out_dg) except +
    double G5(const double rij, const double rik, const double rjk, const double rc, const double eta, const double _lambda, const double xi, const double cos_v, const double* dcos_v, double* out_dg) except + 

def g1(rij, rc):
    cdef double out_dg[3]
    out_dg[:] = [0, 0, 0]
    result = G1(rij, rc, out_dg)
    return result, out_dg

def g2(rij, rc, eta, rs):
    cdef double out_dg[3]
    out_dg[:] = [0, 0, 0]
    result = G2(rij, rc, eta, rs, out_dg)
    return result, out_dg

def g3(rij, rc, k):
    cdef double out_dg[3]
    out_dg[:] = [0, 0, 0]
    result = G3(rij, rc, k, out_dg)
    return result, out_dg

def g4(rij, rik, rjk, rc, eta, _lambda, xi, cos_v, dcos_v_list):
    cdef double out_dg[3], dcos_v[3]
    out_dg[:] = [0, 0, 0]
    dcos_v[:] = dcos_v_list
    result = G4(rij, rik, rjk, rc, eta, _lambda, xi, cos_v, dcos_v, out_dg)
    return result, out_dg

def g5(rij, rik, rjk, rc, eta, _lambda, xi, cos_v, dcos_v_list):
    cdef double out_dg[3], dcos_v[3]
    dcos_v[:] = dcos_v_list
    out_dg[:] = [0, 0, 0]
    result = G5(rij, rik, rjk, rc, eta, _lambda, xi, cos_v, dcos_v, out_dg)
    return result, out_dg