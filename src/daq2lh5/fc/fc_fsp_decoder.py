from __future__ import annotations

import copy
from typing import Any

import lgdo
import numpy as np
from fcio import FCIO, Limits


from ..data_decoder import DataDecoder
from .fc_eventheader_decoder import get_key

fsp_config_decoded_values = {
    "packet_id": {
        "dtype": "uint32",
        "description": "The index of this decoded packet in the file.",
    },
    "buffer_max_states": {"dtype": "int32", "description": ""},
    "buffer_window_nsec": {"dtype": "int64", "description": ""},
    "trg_hwm_min_multiplicity": {"dtype": "int32", "description": ""},
    "trg_hwm_prescale_ratio": {"dtype": "int32", "description": ""},
    "trg_wps_prescale_ratio": {"dtype": "int32", "description": ""},
    "trg_wps_coincident_sum_threshold": {"dtype": "float32", "description": ""},
    "trg_wps_sum_threshold": {"dtype": "float32", "description": ""},
    "trg_wps_prescale_rate": {"dtype": "float32", "description": ""},
    "trg_hwm_prescale_rate": {"dtype": "float32", "description": ""},
    "trg_wps_ref_flags_hwm": {"dtype": "int64", "description": ""},
    "trg_wps_ref_flags_ct": {"dtype": "int64", "description": ""},
    "trg_wps_ref_flags_wps": {"dtype": "int64", "description": ""},
    "trg_wps_ref_map_idx": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "trg_enabled_write_flags_trigger": {"dtype": "int64", "description": ""},
    "trg_enabled_write_flags_event": {"dtype": "int64", "description": ""},
    "trg_pre_trigger_window_nsec": {"dtype": "int64", "description": ""},
    "trg_post_trigger_window_nsec": {"dtype": "int64", "description": ""},
    "dsp_wps_tracemap_format": {"dtype": "int32", "description": ""},
    "dsp_wps_tracemap_indices": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_tracemap_enabled": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_tracemap_label": {
        "dtype": "|S7",
        "datatype": "array_of_equalsized_arrays<1,1>{string}",
        "length": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_gains": {
        "dtype": "float32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_thresholds": {
        "dtype": "float32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_lowpass": {
        "dtype": "float32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_shaping_widths": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_margin_front": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_margin_back": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_start_sample": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_stop_sample": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_wps_dsp_max_margin_front": {"dtype": "int32", "description": ""},
    "dsp_wps_dsp_max_margin_back": {"dtype": "int32", "description": ""},
    "dsp_wps_apply_gain_scaling": {"dtype": "int32", "description": ""},
    "dsp_wps_sum_window_size": {"dtype": "int32", "description": ""},
    "dsp_wps_sum_window_start_sample": {"dtype": "int32", "description": ""},
    "dsp_wps_sum_window_stop_sample": {"dtype": "int32", "description": ""},
    "dsp_wps_sub_event_sum_threshold": {"dtype": "float32", "description": ""},
    "dsp_hwm_tracemap_format": {"dtype": "int32", "description": ""},
    "dsp_hwm_tracemap_indices": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_hwm_tracemap_enabled": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_hwm_tracemap_label": {
        "dtype": "|S7",
        "datatype": "array_of_equalsized_arrays<1,1>{string}",
        "length": Limits.MaxChannels,
        "description": "",
    },
    "dsp_hwm_fpga_energy_threshold_adc": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_ct_tracemap_format": {"dtype": "int32", "description": ""},
    "dsp_ct_tracemap_indices": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_ct_tracemap_enabled": {
        "dtype": "int32",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "dsp_ct_tracemap_label": {
        "dtype": "|S7",
        "datatype": "array_of_equalsized_arrays<1,1>{string}",
        "length": Limits.MaxChannels,
        "description": "",
    },
    "dsp_ct_thresholds": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
}


class FSPConfigDecoder(DataDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fsp_config = lgdo.Struct()

        self.decoded_values = copy.deepcopy(fsp_config_decoded_values)
        self.key_list = []

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:
        self.key_list = [f"fsp_config_{get_key(fcio_stream.config.streamid, 0, 0)}"]
        if fcio_stream.fsp is not None:
            wps_n_traces = fcio_stream.fsp.config.wps["tracemap"]["n_mapped"]
            hwm_n_traces = fcio_stream.fsp.config.hwm["tracemap"]["n_mapped"]
            ct_n_traces = fcio_stream.fsp.config.ct["tracemap"]["n_mapped"]
            self.decoded_values["dsp_wps_tracemap_label"]["length"] = wps_n_traces
            self.decoded_values["dsp_hwm_tracemap_label"]["length"] = hwm_n_traces
            self.decoded_values["dsp_ct_tracemap_label"]["length"] = ct_n_traces

            self.decoded_values["trg_wps_ref_map_idx"]["length_guess"] = (
                fcio_stream.fsp.config.triggerconfig["n_wps_ref_map_idx"]
            )
            self.decoded_values["dsp_wps_tracemap_indices"][
                "length_guess"
            ] = wps_n_traces
            self.decoded_values["dsp_wps_tracemap_enabled"]["length_guess"] = (
                fcio_stream.fsp.config.wps["tracemap"]["n_enabled"]
            )
            self.decoded_values["dsp_wps_gains"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_thresholds"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_lowpass"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_shaping_widths"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_margin_front"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_margin_back"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_start_sample"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_wps_stop_sample"]["length_guess"] = wps_n_traces
            self.decoded_values["dsp_hwm_tracemap_indices"][
                "length_guess"
            ] = hwm_n_traces
            self.decoded_values["dsp_hwm_tracemap_enabled"]["length_guess"] = (
                fcio_stream.fsp.config.hwm["tracemap"]["n_enabled"]
            )
            self.decoded_values["dsp_hwm_fpga_energy_threshold_adc"][
                "length_guess"
            ] = hwm_n_traces
            self.decoded_values["dsp_ct_tracemap_indices"]["length_guess"] = ct_n_traces
            self.decoded_values["dsp_ct_tracemap_enabled"]["length_guess"] = (
                fcio_stream.fsp.config.ct["tracemap"]["n_enabled"]
            )
            self.decoded_values["dsp_ct_thresholds"]["length_guess"] = ct_n_traces

    def get_decoded_values(self, key: int = None) -> dict[str, dict[str, Any]]:
        return self.decoded_values

    def decode_packet(
        self,
        fcio: FCIO,
        fsp_config_rb: lgdo.Table,
        packet_id: int,
    ) -> bool:

        tbl = fsp_config_rb.lgdo
        loc = fsp_config_rb.loc

        # FSP: Buffer
        buffer = fcio.fsp.config.buffer
        # TriggerConfig
        triggerconfig = fcio.fsp.config.triggerconfig
        #  DSPWindowedPeakSum:
        wps = fcio.fsp.config.wps
        wps_n_traces = wps["tracemap"]["n_mapped"]
        # DSPHardwareMajority:
        hwm = fcio.fsp.config.hwm
        hwm_n_traces = hwm["tracemap"]["n_mapped"]
        # DSPChannelThreshold:
        ct = fcio.fsp.config.ct
        ct_n_traces = ct["tracemap"]["n_mapped"]

        tbl["packet_id"].nda[loc] = packet_id

        tbl["buffer_max_states"].nda[loc] = np.int32(buffer["max_states"])
        tbl["buffer_window_nsec"].nda[loc] = np.int64(
            buffer["buffer_window"]["seconds"] * 1e9
            + buffer["buffer_window"]["nanoseconds"]
        )
        tbl["trg_hwm_min_multiplicity"].nda[loc] = np.int32(
            triggerconfig["hwm_min_multiplicity"]
        )
        tbl["trg_hwm_prescale_ratio"].nda[loc] = np.int32(
            triggerconfig["hwm_prescale_ratio"]
        )
        tbl["trg_wps_prescale_ratio"].nda[loc] = np.int32(
            triggerconfig["wps_prescale_ratio"]
        )
        tbl["trg_wps_coincident_sum_threshold"].nda[loc] = np.float32(
            triggerconfig["wps_coincident_sum_threshold"]
        )
        tbl["trg_wps_sum_threshold"].nda[loc] = np.float32(
            triggerconfig["wps_sum_threshold"]
        )
        tbl["trg_wps_prescale_rate"].nda[loc] = np.float32(
            triggerconfig["wps_prescale_rate"]
        )
        tbl["trg_hwm_prescale_rate"].nda[loc] = np.float32(
            triggerconfig["hwm_prescale_rate"]
        )
        tbl["trg_wps_ref_flags_hwm"].nda[loc] = np.int64(
            triggerconfig["wps_ref_flags_hwm"]["is_flagged"]
        )
        tbl["trg_wps_ref_flags_ct"].nda[loc] = np.int64(
            triggerconfig["wps_ref_flags_ct"]["is_flagged"]
        )
        tbl["trg_wps_ref_flags_wps"].nda[loc] = np.int64(
            triggerconfig["wps_ref_flags_wps"]["is_flagged"]
        )
        wps_ref_map_idx = np.array(triggerconfig["wps_ref_map_idx"], dtype="int32")[
            : triggerconfig["n_wps_ref_map_idx"]
        ]
        tbl["trg_wps_ref_map_idx"]._set_vector_unsafe(loc, wps_ref_map_idx)

        tbl["trg_enabled_write_flags_trigger"].nda[loc] = np.int64(
            triggerconfig["enabled_flags"]["trigger"]["is_flagged"]
        )
        tbl["trg_enabled_write_flags_event"].nda[loc] = np.int64(
            triggerconfig["enabled_flags"]["event"]["is_flagged"]
        )
        tbl["trg_pre_trigger_window_nsec"].nda[loc] = np.int64(
            triggerconfig["pre_trigger_window"]["seconds"] * 1e9
            + triggerconfig["pre_trigger_window"]["nanoseconds"]
        )
        tbl["trg_post_trigger_window_nsec"].nda[loc] = np.int64(
            triggerconfig["post_trigger_window"]["seconds"] * 1e9
            + triggerconfig["post_trigger_window"]["nanoseconds"]
        )
        tbl["dsp_wps_tracemap_format"].nda[loc] = np.int32(wps["tracemap"]["format"])
        tbl["dsp_wps_tracemap_indices"]._set_vector_unsafe(
            loc, np.array(wps["tracemap"]["map"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_tracemap_enabled"]._set_vector_unsafe(
            loc,
            np.array(wps["tracemap"]["enabled"], dtype="int32")[
                : wps["tracemap"]["n_enabled"]
            ],
        )

        tbl["dsp_wps_tracemap_label"].nda[loc][:] = np.array(
            wps["tracemap"]["label"], dtype="|S7"
        )

        tbl["dsp_wps_gains"]._set_vector_unsafe(
            loc, np.array(wps["gains"], dtype="float32")[:wps_n_traces]
        )
        tbl["dsp_wps_thresholds"]._set_vector_unsafe(
            loc, np.array(wps["thresholds"], dtype="float32")[:wps_n_traces]
        )
        tbl["dsp_wps_lowpass"]._set_vector_unsafe(
            loc, np.array(wps["lowpass"], dtype="float32")[:wps_n_traces]
        )
        tbl["dsp_wps_shaping_widths"]._set_vector_unsafe(
            loc, np.array(wps["shaping_widths"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_margin_front"]._set_vector_unsafe(
            loc, np.array(wps["dsp_margin_front"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_margin_back"]._set_vector_unsafe(
            loc, np.array(wps["dsp_margin_back"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_start_sample"]._set_vector_unsafe(
            loc, np.array(wps["dsp_start_sample"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_stop_sample"]._set_vector_unsafe(
            loc, np.array(wps["dsp_stop_sample"], dtype="int32")[:wps_n_traces]
        )
        tbl["dsp_wps_dsp_max_margin_front"].nda[loc] = np.int32(
            wps["dsp_max_margin_front"]
        )
        tbl["dsp_wps_dsp_max_margin_back"].nda[loc] = np.int32(
            wps["dsp_max_margin_back"]
        )
        tbl["dsp_wps_apply_gain_scaling"].nda[loc] = np.int32(wps["apply_gain_scaling"])
        tbl["dsp_wps_sum_window_size"].nda[loc] = np.int32(wps["sum_window_size"])
        tbl["dsp_wps_sum_window_start_sample"].nda[loc] = np.int32(
            wps["sum_window_start_sample"]
        )
        tbl["dsp_wps_sum_window_stop_sample"].nda[loc] = np.int32(
            wps["sum_window_stop_sample"]
        )
        tbl["dsp_wps_sub_event_sum_threshold"].nda[loc] = np.float32(
            wps["sub_event_sum_threshold"]
        )
        tbl["dsp_hwm_tracemap_format"].nda[loc] = np.int32(hwm["tracemap"]["format"])
        tbl["dsp_hwm_tracemap_indices"]._set_vector_unsafe(
            loc, np.array(hwm["tracemap"]["map"], dtype="int32")[:hwm_n_traces]
        )
        tbl["dsp_hwm_tracemap_enabled"]._set_vector_unsafe(
            loc,
            np.array(hwm["tracemap"]["enabled"], dtype="int32")[
                : hwm["tracemap"]["n_enabled"]
            ],
        )

        tbl["dsp_hwm_tracemap_label"].nda[loc][:] = np.array(
            hwm["tracemap"]["label"], dtype="|S7"
        )

        tbl["dsp_hwm_fpga_energy_threshold_adc"]._set_vector_unsafe(
            loc,
            np.array(hwm["fpga_energy_threshold_adc"], dtype="uint16")[:hwm_n_traces],
        )
        tbl["dsp_ct_tracemap_format"].nda[loc] = np.int32(ct["tracemap"]["format"])
        tbl["dsp_ct_tracemap_indices"]._set_vector_unsafe(
            loc, np.array(ct["tracemap"]["map"], dtype="int32")[:ct_n_traces]
        )
        tbl["dsp_ct_tracemap_enabled"]._set_vector_unsafe(
            loc,
            np.array(ct["tracemap"]["enabled"], dtype="int32")[
                : ct["tracemap"]["n_enabled"]
            ],
        )

        tbl["dsp_ct_tracemap_label"].nda[loc][:] = np.array(
            ct["tracemap"]["label"], dtype="|S7"
        )

        tbl["dsp_ct_thresholds"]._set_vector_unsafe(
            loc, np.array(ct["thresholds"], dtype="uint16")[:ct_n_traces]
        )

        fsp_config_rb.loc += 1

        return fsp_config_rb.is_full()

    def decode_config(self, fcio: FCIO) -> lgdo.Struct:
        # FSP: Buffer
        buffer = fcio.fsp.config.buffer
        self.fsp_config.add_field(
            "buffer_max_states", lgdo.Scalar(np.int32(buffer["max_states"]))
        )
        self.fsp_config.add_field(
            "buffer_window_nsec",
            lgdo.Scalar(
                np.int64(
                    buffer["buffer_window"]["seconds"] * 1e9
                    + buffer["buffer_window"]["nanoseconds"]
                )
            ),
        )

        # TriggerConfig
        triggerconfig = fcio.fsp.config.triggerconfig
        self.fsp_config.add_field(
            "trg_hwm_min_multiplicity",
            lgdo.Scalar(np.int32(triggerconfig["hwm_min_multiplicity"])),
        )
        self.fsp_config.add_field(
            "trg_hwm_prescale_ratio",
            lgdo.Scalar(np.int32(triggerconfig["hwm_prescale_ratio"])),
        )
        self.fsp_config.add_field(
            "trg_wps_prescale_ratio",
            lgdo.Scalar(np.int32(triggerconfig["wps_prescale_ratio"])),
        )
        self.fsp_config.add_field(
            "trg_wps_coincident_sum_threshold",
            lgdo.Scalar(np.float32(triggerconfig["wps_coincident_sum_threshold"])),
        )
        self.fsp_config.add_field(
            "trg_wps_sum_threshold",
            lgdo.Scalar(np.float32(triggerconfig["wps_sum_threshold"])),
        )
        self.fsp_config.add_field(
            "trg_wps_prescale_rate",
            lgdo.Scalar(np.float32(triggerconfig["wps_prescale_rate"])),
        )
        self.fsp_config.add_field(
            "trg_hwm_prescale_rate",
            lgdo.Scalar(np.float32(triggerconfig["hwm_prescale_rate"])),
        )

        self.fsp_config.add_field(
            "trg_wps_ref_flags_hwm",
            lgdo.Scalar(np.int64(triggerconfig["wps_ref_flags_hwm"]["is_flagged"])),
        )
        self.fsp_config.add_field(
            "trg_wps_ref_flags_ct",
            lgdo.Scalar(np.int64(triggerconfig["wps_ref_flags_ct"]["is_flagged"])),
        )
        self.fsp_config.add_field(
            "trg_wps_ref_flags_wps",
            lgdo.Scalar(np.int64(triggerconfig["wps_ref_flags_wps"]["is_flagged"])),
        )

        wps_ref_map_idx = np.array(triggerconfig["wps_ref_map_idx"], dtype="int32")[
            : triggerconfig["n_wps_ref_map_idx"]
        ]
        self.fsp_config.add_field("trg_wps_ref_map_idx", lgdo.Array(wps_ref_map_idx))
        self.fsp_config.add_field(
            "trg_enabled_write_flags_trigger",
            lgdo.Scalar(
                np.int64(triggerconfig["enabled_flags"]["trigger"]["is_flagged"])
            ),
        )
        self.fsp_config.add_field(
            "trg_enabled_write_flags_event",
            lgdo.Scalar(
                np.int64(triggerconfig["enabled_flags"]["event"]["is_flagged"])
            ),
        )
        self.fsp_config.add_field(
            "trg_pre_trigger_window_nsec",
            lgdo.Scalar(
                np.int64(
                    triggerconfig["pre_trigger_window"]["seconds"] * 1e9
                    + triggerconfig["pre_trigger_window"]["nanoseconds"]
                )
            ),
        )
        self.fsp_config.add_field(
            "trg_post_trigger_window_nsec",
            lgdo.Scalar(
                np.int64(
                    triggerconfig["post_trigger_window"]["seconds"] * 1e9
                    + triggerconfig["post_trigger_window"]["nanoseconds"]
                )
            ),
        )

        #  DSPWindowedPeakSum:
        wps = fcio.fsp.config.wps
        wps_n_traces = wps["tracemap"]["n_mapped"]
        self.fsp_config.add_field(
            "dsp_wps_tracemap_format", lgdo.Scalar(np.int32(wps["tracemap"]["format"]))
        )
        self.fsp_config.add_field(
            "dsp_wps_tracemap_indices",
            lgdo.Array(np.array(wps["tracemap"]["map"], dtype="int32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_tracemap_enabled",
            lgdo.Array(
                np.array(wps["tracemap"]["enabled"], dtype="int32")[
                    : wps["tracemap"]["n_enabled"]
                ]
            ),
        )
        self.fsp_config.add_field(
            "dsp_wps_tracemap_label",
            lgdo.Array(np.array(wps["tracemap"]["label"], dtype="|S7")[:wps_n_traces]),
        )

        self.fsp_config.add_field(
            "dsp_wps_gains",
            lgdo.Array(np.array(wps["gains"], dtype="float32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_thresholds",
            lgdo.Array(np.array(wps["thresholds"], dtype="float32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_lowpass",
            lgdo.Array(np.array(wps["lowpass"], dtype="float32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_shaping_widths",
            lgdo.Array(np.array(wps["shaping_widths"], dtype="int32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_margin_front",
            lgdo.Array(np.array(wps["dsp_margin_front"], dtype="int32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_margin_back",
            lgdo.Array(np.array(wps["dsp_margin_back"], dtype="int32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_start_sample",
            lgdo.Array(np.array(wps["dsp_start_sample"], dtype="int32")[:wps_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_wps_stop_sample",
            lgdo.Array(np.array(wps["dsp_stop_sample"], dtype="int32")[:wps_n_traces]),
        )

        self.fsp_config.add_field(
            "dsp_wps_dsp_max_margin_front",
            lgdo.Scalar(np.int32(wps["dsp_max_margin_front"])),
        )
        self.fsp_config.add_field(
            "dsp_wps_dsp_max_margin_back",
            lgdo.Scalar(np.int32(wps["dsp_max_margin_back"])),
        )
        self.fsp_config.add_field(
            "dsp_wps_apply_gain_scaling",
            lgdo.Scalar(np.int32(wps["apply_gain_scaling"])),
        )
        self.fsp_config.add_field(
            "dsp_wps_sum_window_size", lgdo.Scalar(np.int32(wps["sum_window_size"]))
        )
        self.fsp_config.add_field(
            "dsp_wps_sum_window_start_sample",
            lgdo.Scalar(np.int32(wps["sum_window_start_sample"])),
        )
        self.fsp_config.add_field(
            "dsp_wps_sum_window_stop_sample",
            lgdo.Scalar(np.int32(wps["sum_window_stop_sample"])),
        )
        self.fsp_config.add_field(
            "dsp_wps_sub_event_sum_threshold",
            lgdo.Scalar(np.float32(wps["sub_event_sum_threshold"])),
        )

        # DSPHardwareMajority:
        hwm = fcio.fsp.config.hwm
        hwm_n_traces = hwm["tracemap"]["n_mapped"]
        self.fsp_config.add_field(
            "dsp_hwm_tracemap_format", lgdo.Scalar(np.int32(hwm["tracemap"]["format"]))
        )
        self.fsp_config.add_field(
            "dsp_hwm_tracemap_indices",
            lgdo.Array(np.array(hwm["tracemap"]["map"], dtype="int32")[:hwm_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_hwm_tracemap_enabled",
            lgdo.Array(
                np.array(hwm["tracemap"]["enabled"], dtype="int32")[
                    : hwm["tracemap"]["n_enabled"]
                ]
            ),
        )
        self.fsp_config.add_field(
            "dsp_hwm_tracemap_label",
            lgdo.Array(np.array(hwm["tracemap"]["label"], dtype="|S7")[:hwm_n_traces]),
        )

        self.fsp_config.add_field(
            "dsp_hwm_fpga_energy_threshold_adc",
            lgdo.Array(
                np.array(hwm["fpga_energy_threshold_adc"], dtype="uint16")[
                    :hwm_n_traces
                ]
            ),
        )

        # DSPChannelThreshold:
        ct = fcio.fsp.config.ct
        ct_n_traces = ct["tracemap"]["n_mapped"]
        self.fsp_config.add_field(
            "dsp_ct_tracemap_format", lgdo.Scalar(np.int32(ct["tracemap"]["format"]))
        )
        self.fsp_config.add_field(
            "dsp_ct_tracemap_indices",
            lgdo.Array(np.array(ct["tracemap"]["map"], dtype="int32")[:ct_n_traces]),
        )
        self.fsp_config.add_field(
            "dsp_ct_tracemap_enabled",
            lgdo.Array(
                np.array(ct["tracemap"]["enabled"], dtype="int32")[
                    : ct["tracemap"]["n_enabled"]
                ]
            ),
        )
        self.fsp_config.add_field(
            "dsp_ct_tracemap_label",
            lgdo.Array(np.array(ct["tracemap"]["label"], dtype="|S7")[:ct_n_traces]),
        )

        self.fsp_config.add_field(
            "dsp_ct_thresholds",
            lgdo.Array(np.array(ct["thresholds"], dtype="uint16")[:ct_n_traces]),
        )
        return self.fsp_config

    def make_lgdo(self, key: int | str = None, size: int = None) -> lgdo.Struct:
        return self.fsp_config


fsp_status_decoded_values = {
    "packet_id": {"dtype": "uint32"},
    "start_time": {"dtype": "float64"},
    "log_time": {"dtype": "float64"},
    "dt_logtime": {"dtype": "float64"},
    "runtime": {"dtype": "float64"},
    "n_read_events": {"dtype": "int32"},
    "n_written_events": {"dtype": "int32"},
    "n_discarded_events": {"dtype": "int32"},
    "dt_n_read_events": {"dtype": "int32"},
    "dt_n_written_events": {"dtype": "int32"},
    "dt_n_discarded_events": {"dtype": "int32"},
    "dt": {"dtype": "float64"},
    "dt_rate_read_events": {"dtype": "float64"},
    "dt_rate_write_events": {"dtype": "float64"},
    "dt_rate_discard_events": {"dtype": "float64"},
    "avg_rate_read_events": {"dtype": "float64"},
    "avg_rate_write_events": {"dtype": "float64"},
    "avg_rate_discard_events": {"dtype": "float64"},
}


class FSPStatusDecoder(DataDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoded_values = copy.deepcopy(fsp_status_decoded_values)
        self.key_list = []

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:
        self.key_list = [f"fsp_status_{get_key(fcio_stream.config.streamid, 0, 0)}"]

    def get_key_lists(self) -> list[list[int | str]]:
        return [copy.deepcopy(self.key_list)]

    def get_decoded_values(self, key: int = None) -> dict[str, dict[str, Any]]:
        return self.decoded_values

    def decode_packet(
        self,
        fcio: FCIO,
        fsp_status_rbkd: dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:

        key = f"fsp_status_{get_key(fcio.config.streamid, 0, 0)}"
        if key not in fsp_status_rbkd:
            return False
        fsp_status_rb = fsp_status_rbkd[key]

        tbl = fsp_status_rb.lgdo
        loc = fsp_status_rb.loc

        tbl["packet_id"].nda[loc] = packet_id
        tbl["start_time"].nda[loc] = fcio.fsp.status.start_time
        tbl["log_time"].nda[loc] = fcio.fsp.status.log_time
        tbl["dt_logtime"].nda[loc] = fcio.fsp.status.dt_logtime
        tbl["runtime"].nda[loc] = fcio.fsp.status.runtime
        tbl["n_read_events"].nda[loc] = fcio.fsp.status.n_read_events
        tbl["n_written_events"].nda[loc] = fcio.fsp.status.n_written_events
        tbl["n_discarded_events"].nda[loc] = fcio.fsp.status.n_discarded_events
        tbl["dt_n_read_events"].nda[loc] = fcio.fsp.status.dt_n_read_events
        tbl["dt_n_written_events"].nda[loc] = fcio.fsp.status.dt_n_written_events
        tbl["dt_n_discarded_events"].nda[loc] = fcio.fsp.status.dt_n_discarded_events
        tbl["dt"].nda[loc] = fcio.fsp.status.dt
        tbl["dt_rate_read_events"].nda[loc] = fcio.fsp.status.dt_rate_read_events
        tbl["dt_rate_write_events"].nda[loc] = fcio.fsp.status.dt_rate_write_events
        tbl["dt_rate_discard_events"].nda[loc] = fcio.fsp.status.dt_rate_discard_events
        tbl["avg_rate_read_events"].nda[loc] = fcio.fsp.status.avg_rate_read_events
        tbl["avg_rate_write_events"].nda[loc] = fcio.fsp.status.avg_rate_write_events
        tbl["avg_rate_discard_events"].nda[
            loc
        ] = fcio.fsp.status.avg_rate_discard_events

        fsp_status_rb.loc += 1

        return fsp_status_rb.is_full()


fsp_event_decoded_values = {
    "packet_id": {"dtype": "uint32", "description": ""},
    "is_written": {"dtype": "bool", "description": ""},
    "is_extended": {"dtype": "bool", "description": ""},
    "is_consecutive": {"dtype": "bool", "description": ""},
    "is_hwm_prescaled": {"dtype": "bool", "description": ""},
    "is_hwm_multiplicity": {"dtype": "bool", "description": ""},
    "is_wps_sum": {"dtype": "bool", "description": ""},
    "is_wps_coincident_sum": {"dtype": "bool", "description": ""},
    "is_wps_prescaled": {"dtype": "bool", "description": ""},
    "is_ct_multiplicity": {"dtype": "bool", "description": ""},
    "obs_wps_sum_value": {"dtype": "float32", "description": ""},
    "obs_wps_sum_offset": {"dtype": "uint16", "description": ""},
    "obs_wps_sum_multiplicity": {"dtype": "uint16", "description": ""},
    "obs_wps_max_peak_value": {"dtype": "float32", "description": ""},
    "obs_wps_max_peak_offset": {"dtype": "uint16", "description": ""},
    "obs_hwm_multiplicity": {"dtype": "uint16", "description": ""},
    "obs_hwm_max_value": {"dtype": "uint16", "description": ""},
    "obs_hwm_min_value": {"dtype": "uint16", "description": ""},
    "obs_ct_multiplicity": {"dtype": "uint32", "description": ""},
    "obs_ct_trace_idx": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "obs_ct_max": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "",
    },
    "obs_evt_nconsecutive": {"dtype": "int32", "description": ""},
}


class FSPEventDecoder(DataDecoder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoded_values = copy.deepcopy(fsp_event_decoded_values)
        self.key_list = []

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:

        self.key_list = [f"fsp_event_{get_key(fcio_stream.config.streamid, 0, 0)}"]

    def get_decoded_values(self, key: int = None) -> dict[str, dict[str, Any]]:
        return self.decoded_values

    def get_key_lists(self) -> list[list[int | str]]:
        return [copy.deepcopy(self.key_list)]

    def decode_packet(
        self,
        fcio: FCIO,
        fsp_evt_rbkd: dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:

        key = f"fsp_event_{get_key(fcio.config.streamid, 0, 0)}"
        if key not in fsp_evt_rbkd:
            return False
        fsp_evt_rb = fsp_evt_rbkd[key]

        tbl = fsp_evt_rb.lgdo
        loc = fsp_evt_rb.loc

        tbl["packet_id"].nda[loc] = packet_id
        tbl["is_written"].nda[loc] = fcio.fsp.event.is_written
        tbl["is_extended"].nda[loc] = fcio.fsp.event.is_extended
        tbl["is_consecutive"].nda[loc] = fcio.fsp.event.is_consecutive
        tbl["is_hwm_prescaled"].nda[loc] = fcio.fsp.event.is_hwm_prescaled
        tbl["is_hwm_multiplicity"].nda[loc] = fcio.fsp.event.is_hwm_multiplicity
        tbl["is_wps_sum"].nda[loc] = fcio.fsp.event.is_wps_sum
        tbl["is_wps_coincident_sum"].nda[loc] = fcio.fsp.event.is_wps_coincident_sum
        tbl["is_wps_prescaled"].nda[loc] = fcio.fsp.event.is_wps_prescaled
        tbl["is_ct_multiplicity"].nda[loc] = fcio.fsp.event.is_ct_multiplicity

        tbl["obs_wps_sum_value"].nda[loc] = fcio.fsp.event.obs_wps_sum_value
        tbl["obs_wps_sum_offset"].nda[loc] = fcio.fsp.event.obs_wps_sum_offset
        tbl["obs_wps_sum_multiplicity"].nda[
            loc
        ] = fcio.fsp.event.obs_wps_sum_multiplicity
        tbl["obs_wps_max_peak_value"].nda[
            loc
        ] = fcio.fsp.event.obs_wps_max_single_peak_value
        tbl["obs_wps_max_peak_offset"].nda[
            loc
        ] = fcio.fsp.event.obs_wps_max_single_peak_offset
        tbl["obs_hwm_multiplicity"].nda[loc] = fcio.fsp.event.obs_hwm_multiplicity
        tbl["obs_hwm_max_value"].nda[loc] = fcio.fsp.event.obs_hwm_max_value
        tbl["obs_hwm_min_value"].nda[loc] = fcio.fsp.event.obs_hwm_min_value
        tbl["obs_evt_nconsecutive"].nda[loc] = fcio.fsp.event.obs_evt_nconsecutive
        tbl["obs_ct_multiplicity"].nda[loc] = fcio.fsp.event.obs_ct_multiplicity

        lens = fcio.fsp.event.obs_ct_multiplicity
        start = 0 if loc == 0 else tbl["obs_ct_trace_idx"].cumulative_length[loc - 1]
        end = start + lens
        if lens > 0:
            tbl["obs_ct_trace_idx"].flattened_data.nda[
                start:end
            ] = fcio.fsp.event.obs_ct_trace_idx
            tbl["obs_ct_max"].flattened_data.nda[start:end] = fcio.fsp.event.obs_ct_max

        tbl["obs_ct_trace_idx"].cumulative_length[loc] = end
        tbl["obs_ct_max"].cumulative_length[loc] = end

        fsp_evt_rb.loc += 1

        return fsp_evt_rb.is_full()
