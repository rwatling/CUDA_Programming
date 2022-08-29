#!bash
# NEEDS ENV VARIABLES
nvcc ./powercap_basic.c -I/${PAPI_DIR}/include -I/${PAPI_DIR}/share/papi/testlib -L/${PAPI_DIR}/lib -o powercap_basic -lpapi