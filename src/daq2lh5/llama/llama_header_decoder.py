from __future__ import annotations

import logging
import io
import numpy as np
from typing import Dict, Any

import lgdo

from ..data_decoder import DataDecoder

log = logging.getLogger(__name__)


class LLAMAHeaderDecoder(DataDecoder):  # DataDecoder currently unused
    """
    Decode llamaDAQ header data. Includes the file header as well as all available ("open") channel configurations.
    """

    @staticmethod
    def magic_bytes() -> int:
        return 0x4972414c

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = lgdo.Struct()
        self.channel_configs = None
        self.verbose = True ### debug

    def decode_header(self, f_in: io.BufferedReader) -> lgdo.Struct:
        n_bytes_read = 0

        f_in.seek(0)    #should be there anyhow, but re-set if not
        header = f_in.read(16) #read 16 bytes
        n_bytes_read += 16
        evt_data_32 = np.fromstring(header, dtype=np.uint32)
        evt_data_16 = np.fromstring(header, dtype=np.uint16)

        #line0: magic bytes
        magic = evt_data_32[0]
        print(hex(magic))
        if magic == self.magic_bytes():
            if self.verbose > 0:
                print ("Read in file as llamaDAQ-SIS3316, magic bytes correct.")
        else:
            print ("ERROR: Magic bytes not matching for llamaDAQ file!")
            raise RuntimeError("wrong file type")
    
        self.version_major = evt_data_16[4]
        self.version_minor = evt_data_16[3]
        self.version_patch = evt_data_16[2]
        self.length_econf = evt_data_16[5]
        self.number_chOpen = evt_data_32[3]
        
        if self.verbose > 0:
            print ("File version: {}.{}.{}".format(self.version_major, self.version_minor, self.version_patch))
            print ("{} channels open, each config {} bytes long".format(self.number_chOpen, self.length_econf))

        n_bytes_read += self.__decode_channelConfigs(f_in)

        print(self.channel_configs[0][0]["MAW3_offset"])

        # assemble LGDO struct:
        self.config.add_field("version_major", self.version_major)
        self.config.add_field("version_minor", self.version_minor)
        self.config.add_field("version_patch", self.version_patch)
        self.config.add_field("length_econf", self.length_econf)
        self.config.add_field("number_chOpen", self.number_chOpen)
        
        for fadcid, fadc in self.channel_configs.items():
            fadc_lgdo = lgdo.Struct()
            for chid, ch in fadc.items():
                ch_lgdo = lgdo.Struct()
                for key, value in ch.items():
                    ch_lgdo.add_field(key, value)
                fadc_lgdo.add_field("ch_{:02d}".format(chid), ch_lgdo)
            self.config.add_field("fadc_{:02d}".format(fadcid), fadc_lgdo)


        return self.config, n_bytes_read
    
    #override from DataDecoder
    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        return self.config
    

    def __decode_channelConfigs(self, f_in: io.BufferedReader) -> int:
        """
        Reads the metadata from the beginning of the file (the "channel configuration" part, directly after the file header).
        Creates a dictionary of the metadata for each FADC/channel combination, which is returned
        
        structure of channelConfigs:
        FADCindex      channelIndex
        A ------------- x ----------- metadata for FADC A channel x
                      | y ----------- metadata for FADC A channel y
                      | z ----------- metadata for FADC A channel z
                      
        B ------------- k ----------- metadata for FADC B channel k
                      | l ----------- metadata for FADC B channel l
        ...
                      
        returns number of bytes read
        """
        #f_in.seek(16)    #should be after file header anyhow, but re-set if not
        n_bytes_read = 0
        self.channel_configs = {}

        if self.length_econf != 88:
            raise RuntimeError("Invalid channel configuration format")

        for i in range(0, self.number_chOpen):
            if self.verbose > 1:
                print("reading in channel config {}".format(i))
                
            channel = f_in.read(self.length_econf)
            n_bytes_read += self.length_econf
            ch_dpf = channel[16:32]
            evt_data_32 = np.fromstring(channel, dtype=np.uint32)
            evt_data_dpf = np.fromstring(ch_dpf, dtype=np.float64)
            
            fadcIndex = evt_data_32[0]
            channelIndex = evt_data_32[1]
            
            if fadcIndex in self.channel_configs:
                #print("pre-existing fadc")
                pass
            else:
                #print("new fadc #{}".format(fadcIndex))
                self.channel_configs[fadcIndex] = {}
    
            if channelIndex in self.channel_configs[fadcIndex]:
                raise RuntimeError("duplicate channel configuration in file: FADCID: {}, ChannelID: {}".format(fadcIndex, channelIndex))
            else:
                self.channel_configs[fadcIndex][channelIndex] = {}
                
            self.channel_configs[fadcIndex][channelIndex]["14BitFlag"] = evt_data_32[2] & 0x00000001
            if evt_data_32[2] & 0x00000002 == 0:
                print("WARNING: Channel in configuration marked as non-open!")
            self.channel_configs[fadcIndex][channelIndex]["ADC_offset"] = evt_data_32[3]
            self.channel_configs[fadcIndex][channelIndex]["sample_freq"] = evt_data_dpf[0]     #64 bit float
            self.channel_configs[fadcIndex][channelIndex]["gain"] = evt_data_dpf[1]
            self.channel_configs[fadcIndex][channelIndex]["format_bits"] = evt_data_32[8]
            self.channel_configs[fadcIndex][channelIndex]["sample_start_index"] = evt_data_32[9]
            self.channel_configs[fadcIndex][channelIndex]["sample_pretrigger"] = evt_data_32[10]
            self.channel_configs[fadcIndex][channelIndex]["avg_sample_pretrigger"] = evt_data_32[11]
            self.channel_configs[fadcIndex][channelIndex]["avg_mode"] = evt_data_32[12]
            self.channel_configs[fadcIndex][channelIndex]["sample_length"] = evt_data_32[13]
            self.channel_configs[fadcIndex][channelIndex]["avg_sample_length"] = evt_data_32[14]
            self.channel_configs[fadcIndex][channelIndex]["MAW_buffer_length"] = evt_data_32[15]
            self.channel_configs[fadcIndex][channelIndex]["event_length"] = evt_data_32[16]
            self.channel_configs[fadcIndex][channelIndex]["event_header_length"] = evt_data_32[17]
            self.channel_configs[fadcIndex][channelIndex]["accum6_offset"] = evt_data_32[18]
            self.channel_configs[fadcIndex][channelIndex]["accum2_offset"] = evt_data_32[19]
            self.channel_configs[fadcIndex][channelIndex]["MAW3_offset"] = evt_data_32[20]
            self.channel_configs[fadcIndex][channelIndex]["energy_offset"] = evt_data_32[21]

        return n_bytes_read


            