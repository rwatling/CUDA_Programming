#!bash
nvcc ./src/hello_world.cpp -I/${PAPI_DIR}/include -I/${PAPI_DIR}/share/papi/testlib -L/${PAPI_DIR}/lib -o powercap_basic -lpapi