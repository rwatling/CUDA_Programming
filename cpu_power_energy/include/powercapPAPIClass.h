#ifndef POWERCAPPAPICLASS_H_
#define POWERCAPPAPICLASS_H_ 1

/* A restructuring of the code found in 
/papi-install/share/papi/components/powercap/powercap_basic.c
for use in CPU powermeasurements */

#include "papi.h"
#include "papi_test.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <stdio.h>
#include <string.h>

#define MAX_powercap_EVENTS 64

using namespace std;

class powercapPAPIClass {
    public:
        powercapPAPIClass(std::string &filename):
        filename_ { filename }, EventSet { PAPI_NULL },
        evinfo { }, num_events { }, values { } {

            outfile_.open(filename_, std::ios::out);
        }

        void initialize_papi() {
            int numcmp;
            int cid;
            int powercap_cid=-1;
            int code;
            int search;
            int retval;

            const PAPI_component_info_t *cmpinfo = NULL;

            /* PAPI Initialization */
            retval = PAPI_library_init( PAPI_VER_CURRENT );
            if ( retval != PAPI_VER_CURRENT )
                outfile_ << "PAPI_library_init failed\n";

            numcmp = PAPI_num_components();

            for( cid=0; cid<numcmp; cid++ ) {

                if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
                    outfile_ << "PAPI_get_component_info failed\n";

                if ( strstr( cmpinfo->name,"powercap" ) ) {
                    powercap_cid=cid;

                    if ( cmpinfo->disabled ) {
                            outfile_ << "powercap component disabled: " <<
                                    cmpinfo->disabled_reason << endl;
                    }
                    break;
                }
            }

            /* Component not found */
            if ( cid==numcmp )
                outfile_ << "No powercap component found\n";

            /* Skip if component has no counters */
            if ( cmpinfo->num_cntrs==0 )
                outfile_ << "No counters in the powercap component\n";

            /* Create EventSet */
            retval = PAPI_create_eventset( &EventSet );
            if ( retval != PAPI_OK )
                outfile_ << "PAPI_create_eventset() failed\n";

            /* Add all events */
            code = PAPI_NATIVE_MASK;
            search = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, powercap_cid );
            
            while ( search == PAPI_OK ) {
                
                retval = PAPI_event_code_to_name( code, event_names[num_events] );
                if ( retval != PAPI_OK )
                    outfile_ << "Error from PAPI_event_code_to_name\n";

                retval = PAPI_get_event_info( code,&evinfo );
                if ( retval != PAPI_OK )
                    outfile_ << "Error getting event info\n";

                strncpy( event_descrs[num_events],evinfo.long_descr,sizeof( event_descrs[0] )-1 );
                strncpy( units[num_events],evinfo.units,sizeof( units[0] )-1 );
                
                // buffer must be null terminated to safely use strstr operation on it below
                units[num_events][sizeof( units[0] )-1] = '\0';
                data_type[num_events] = evinfo.data_type;
                retval = PAPI_add_event( EventSet, code );

                if ( retval != PAPI_OK )
                    break; /* We've hit an event limit */
                num_events++;

                search = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, powercap_cid );
            }

            values = (long long *) calloc( num_events, sizeof( long long ) );
            if ( values==NULL )
                outfile_ << "calloc for PAPI values failed\n";

            outfile_ << "PAPI initialized\n";

        }

        void start_counting( ) {
            int retval;

            start_time = PAPI_get_real_nsec();
            retval = PAPI_start( EventSet );
            if ( retval != PAPI_OK )
                outfile_ << "PAPI failed @ start_counting()\n";
        }

        void stop_counting( ) {
            end_time = PAPI_get_real_nsec();
            int retval = PAPI_stop( EventSet, values );

            if ( retval != PAPI_OK )
                outfile_ << "PAPI failed @ stop_counting() \n";

            elapsed_time=( ( double )( end_time-start_time ) )/1.0e9;

            report_out();

            outfile_.close();

        }

        void finalize_papi() {
            int retval;

            /* Done, clean up */
            retval = PAPI_cleanup_eventset( EventSet );
            if ( retval != PAPI_OK )
               outfile_ << "PAPI_cleanup_eventset() failed\n";

            retval = PAPI_destroy_eventset( &EventSet );
            if ( retval != PAPI_OK )
                outfile_ << "PAPI_destroy_eventset() failed\n";

        }

    private:
        int EventSet;
        int num_events;
        time_t start_time;
        time_t end_time;
        time_t elapsed_time;
        long long *values;
        char event_names[MAX_powercap_EVENTS][PAPI_MAX_STR_LEN];
        char event_descrs[MAX_powercap_EVENTS][PAPI_MAX_STR_LEN];
        char units[MAX_powercap_EVENTS][PAPI_MIN_STR_LEN];
        int data_type[MAX_powercap_EVENTS];
        std::string filename_;
        std::ofstream outfile_;
        PAPI_event_info_t evinfo;


        void report_out( ) {
            int i = 0;

            // Write out information
            outfile_ << "\nStopping measurements, took %.3fs, gathering results...\n\n", elapsed_time;

            outfile_ << "\n";
            outfile_ << "scaled energy measurements:\n";
            for( i=0; i<num_events; i++ ) {
                if ( strstr( event_names[i],"ENERGY_UJ" ) ) {
                    if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                        outfile_ << event_names[i] << "\t" << event_descrs[i] 
                                 << " " << (double) values[i] / 1.0e6 << " J\tAverage Power " 
                                 << ( (double) values[i]/1.0e6) / elapsed_time << " W\n";

                        /*outfile_ << "%-45s%-20s%4.6f J (Average Power %.1fW)\n",
                                event_names[i], event_descrs[i],
                                ( double )values[i]/1.0e6,
                                ( ( double )values[i]/1.0e6 )/elapsed_time;*/
                    }
                }
            }

            outfile_ <<"\n";
            outfile_ << "long term time window values:\n";
            for( i=0; i<num_events; i++ ) {
                if ( strstr( event_names[i], "TIME_WINDOW_A_US" ) ) {
                    if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                        outfile_ << event_names[i] << " "
                        << event_descrs[i] << " " 
                        << ( double ) values[i]/1.0e6 << " (secs)\n";
                        /*outfile_ <<"%-45s%-20s%4f (secs)\n",
                                event_names[i], event_descrs[i],
                                ( double )values[i]/1.0e6;*/
                    }
                }
            }

            outfile_ << "\n";
            outfile_ << "short term time window values:\n";
            for( i=0; i<num_events; i++ ) {
                if ( strstr( event_names[i],"TIME_WINDOW_B_US" ) ) {
                    if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                        outfile_ << event_names[i] << " " 
                                 << event_descrs[i] << " "
                                 << ( double ) values[i]/1.0e6 << " (secs)\n";
                        /*outfile_ << "%-45s%-20s%4f (secs)\n",
                                event_names[i], event_descrs[i],
                                ( double )values[i]/1.0e6;*/
                    }
                }
            }

            outfile_ <<"\n";
            outfile_ << "long term power limit:\n";
            for( i=0; i<num_events; i++ ) {
                if ( strstr( event_names[i], "POWER_LIMIT_A_UW" ) ) {
                    if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                        
                        outfile_ << event_names[i] << " "
                                 << event_descrs[i] << " "
                                 << ( double ) values[i] / 1.0e6 << " (watts)\n";
                        /*outfile_ << "%-45s%-20s%4f (watts)\n",
                                event_names[i], event_descrs[i],
                                ( double )values[i]/1.0e6;*/
                    }
                }
            }

            outfile_ << "\n";
            outfile_ << "short term power limit:\n";
            for( i=0; i<num_events; i++ ) {
                if ( strstr( event_names[i], "POWER_LIMIT_B_UW" ) ) {
                    if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                        
                        outfile_ << event_names[i] << " "
                                 << event_descrs[i] << " "
                                 << ( double ) values[i] / 1.0e6 << " (watts)\n";
                        /*outfile_ << "%-45s%-20s%4f (watts)\n",
                                event_names[i], event_descrs[i],
                                ( double ) values[i]/1.0e6;*/
                    }
                }
            }
        }
};

#endif