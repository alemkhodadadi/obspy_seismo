# -*- coding: utf-8 -*-
"""
SEG Y bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings
from copy import deepcopy
from struct import unpack

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.core.util import open_bytes_stream
from .header import (BINARY_FILE_HEADER_FORMAT, DATA_SAMPLE_FORMAT_CODE_DTYPE,
                     ENDIAN, TRACE_HEADER_FORMAT, TRACE_HEADER_KEYS)
from .segy import _read_segy as _read_segyrev1
from .segy import _read_su as _read_su_file
from .segy import (SEGYBinaryFileHeader, SEGYError, SEGYFile, SEGYTrace,
                   SEGYTraceHeader, SUFile,
                   autodetect_endian_and_sanity_check_su)
from .util import unpack_header_value


# Valid data format codes as specified in the SEGY rev1 manual.
VALID_FORMATS = [1, 2, 3, 4, 5, 8]

# This is the maximum possible interval between two samples due to the nature
# of the SEG Y format.
MAX_INTERVAL_IN_SECONDS = 0.065535

# largest number possible with int16
MAX_NUMBER_OF_SAMPLES = 32767


class SEGYCoreWritingError(SEGYError):
    """
    Raised if the writing of the Stream object fails due to some reason.
    """
    pass


class SEGYSampleIntervalError(SEGYError):
    """
    Raised if the interval between two samples is too large.
    """
    pass


def _is_segy(file):
    """
    Checks whether or not the given file is a SEG Y file.

    :type file: str or file-like object
    :param file: SEG Y file to be checked.
    :rtype: bool
    :return: ``True`` if a SEG Y file.
    """
    # This is a very weak test. It tests two things: First if the data sample
    # format code is valid. This is also used to determine the endianness. This
    # is then used to check if the sampling interval is set to any sane number
    # greater than 0 and that the number of samples per trace is greater than
    # 0.
    try:
        with open_bytes_stream(file) as fp:
            fp.seek(3212)
            _number_of_data_traces = fp.read(2)
            _number_of_auxiliary_traces = fp.read(2)
            _sample_interval = fp.read(2)
            fp.seek(2, 1)
            _samples_per_trace = fp.read(2)
            fp.seek(2, 1)
            data_format_code = fp.read(2)
            fp.seek(3500, 0)
            _format_number = fp.read(2)
            _fixed_length = fp.read(2)
            _extended_number = fp.read(2)
    except Exception:
        return False
    # Unpack using big endian first and check if it is valid.
    try:
        format = unpack(b'>h', data_format_code)[0]
    except Exception:
        return False
    if format in VALID_FORMATS:
        _endian = '>'
    # It can only every be one. It is impossible for little and big endian to
    # both yield a valid data sample format code because they are restricted to
    # be between 1 and 8.
    else:
        format = unpack(b'<h', data_format_code)[0]
        if format in VALID_FORMATS:
            _endian = '<'
        else:
            return False
    # Check if the sample interval and samples per Trace make sense.
    fmt = ('%sh' % _endian).encode('ascii', 'strict')
    _sample_interval = unpack(fmt, _sample_interval)[0]
    _samples_per_trace = unpack(fmt, _samples_per_trace)[0]
    _number_of_data_traces = unpack(fmt, _number_of_data_traces)[0]
    _number_of_auxiliary_traces = unpack(fmt,
                                         _number_of_auxiliary_traces)[0]

    _format_number = unpack(fmt, _format_number)[0]
    # Test the version number. The only really supported version number in
    # ObsPy is 1.0 which is encoded as 0100_16. Many file have version
    # number zero which is used to indicate "traditional SEG-Y" conforming
    # to the 1975 standard.
    # Also allow 0010_16 and 0001_16 as the definition is honestly awkward
    # and I image many writers get it wrong.
    if _format_number not in (0x0000, 0x0100, 0x0010, 0x0001):
        return False

    _fixed_length = unpack(fmt, _fixed_length)[0]
    _extended_number = unpack(fmt, _extended_number)[0]
    # Make some sanity checks and return False if they fail.
    if _sample_interval <= 0 or _samples_per_trace <= 0 \
            or _number_of_data_traces < 0 or _number_of_auxiliary_traces < 0 \
            or _fixed_length < 0 or _extended_number < 0:
        return False
    return True


def _read_segy(filename, headonly=False, byteorder=None,
               textual_header_encoding=None, unpack_trace_headers=False, trace_duration:int=None,
               **kwargs):  # @UnusedVariable
    """
    Reads a SEG Y file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SEG Y rev1 file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the header and omit the waveform
        data.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If it is ``None``, it will try
        to autodetect the endianness. The endianness is always valid for the
        whole file. Defaults to ``None``.
    :type textual_header_encoding: str or ``None``
    :param textual_header_encoding: The encoding of the textual header. Can be
        ``'EBCDIC'``, ``'ASCII'`` or ``None``. If it is ``None``, autodetection
        will be attempted. Defaults to ``None``.
    :type unpack_trace_headers: bool, optional
    :param unpack_trace_headers: Determines whether or not all trace header
        values will be unpacked during reading. If ``False`` it will greatly
        enhance performance and especially memory usage with large files. The
        header values can still be accessed and will be calculated on the fly
        but tab completion will no longer work. Look in the headers.py for a
        list of all possible trace header values. Defaults to ``False``.
    :returns: A ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/00001034.sgy_first_trace")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    Seq. No. in line:    1 | 2009-06-22T14:47:37.000000Z - ... 2001 samples
    """
    # Read file to the internal segy representation.
    segy_object = _read_segyrev1(
        filename, endian=byteorder,
        textual_header_encoding=textual_header_encoding,
        unpack_headers=unpack_trace_headers,
        trace_duration=trace_duration)
    # Create the stream object.
    stream = Stream()
    # SEGY has several file headers that apply to all traces. They will be
    # stored in Stream.stats.
    stream.stats = AttribDict()
    # Get the textual file header.
    textual_file_header = segy_object.textual_file_header
    # The binary file header will be a new AttribDict
    binary_file_header = AttribDict()
    for key, value in segy_object.binary_file_header.__dict__.items():
        setattr(binary_file_header, key, value)
    # Get the data encoding and the endianness from the first trace.
    data_encoding = segy_object.traces[0].data_encoding
    endian = segy_object.traces[0].endian
    textual_file_header_encoding = segy_object.textual_header_encoding.upper()
    # Add the file wide headers.
    stream.stats.textual_file_header = textual_file_header
    stream.stats.binary_file_header = binary_file_header
    # Also set the data encoding, endianness and the encoding of the
    # textual_file_header.
    stream.stats.data_encoding = data_encoding
    stream.stats.endian = endian
    stream.stats.textual_file_header_encoding = \
        textual_file_header_encoding

    # Convert traces to ObsPy Trace objects.
    for tr in segy_object.traces:
        stream.append(tr.to_obspy_trace(
            headonly=headonly,
            unpack_trace_headers=unpack_trace_headers))

    return stream


def seismo_read_segy_(filename, headonly=False, byteorder=None,
               textual_header_encoding=None, unpack_trace_headers=False, trace_duration:int=None,
               **kwargs):  # @UnusedVariable
    """
    This function if a copy of _read_segy_function of obspy.
    all the input arguments are the same except trace_duration. 

    :type trace_duration: int(microseconds), optional 
    :param trace_duration: the length of each trace in microseconds
    
    Reads a SEG Y file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SEG Y rev1 file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the header and omit the waveform
        data.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If it is ``None``, it will try
        to autodetect the endianness. The endianness is always valid for the
        whole file. Defaults to ``None``.
    :type textual_header_encoding: str or ``None``
    :param textual_header_encoding: The encoding of the textual header. Can be
        ``'EBCDIC'``, ``'ASCII'`` or ``None``. If it is ``None``, autodetection
        will be attempted. Defaults to ``None``.
    :type unpack_trace_headers: bool, optional
    :param unpack_trace_headers: Determines whether or not all trace header
        values will be unpacked during reading. If ``False`` it will greatly
        enhance performance and especially memory usage with large files. The
        header values can still be accessed and will be calculated on the fly
        but tab completion will no longer work. Look in the headers.py for a
        list of all possible trace header values. Defaults to ``False``.
    :returns: A an array of 3 ObsPy :class:`~obspy.core.stream.Stream` objects.
        the streams are for components N, E and Z respectively

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/00001034.sgy_first_trace")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    Seq. No. in line:    1 | 2009-06-22T14:47:37.000000Z - ... 2001 samples
    """
    # Read file to the internal segy representation.
    segy_object = _read_segyrev1(
        filename, endian=byteorder,
        textual_header_encoding=textual_header_encoding,
        unpack_headers=unpack_trace_headers,
        trace_duration=trace_duration)
    # Create the stream object.
    stream = Stream()
    # SEGY has several file headers that apply to all traces. They will be
    # stored in Stream.stats.
    stream.stats = AttribDict()
    # Get the textual file header.
    textual_file_header = segy_object.textual_file_header
    # The binary file header will be a new AttribDict
    binary_file_header = AttribDict()
    for key, value in segy_object.binary_file_header.__dict__.items():
        setattr(binary_file_header, key, value)
    # Get the data encoding and the endianness from the first trace.
    data_encoding = segy_object.traces[0].data_encoding
    endian = segy_object.traces[0].endian
    textual_file_header_encoding = segy_object.textual_header_encoding.upper()
    # Add the file wide headers.
    stream.stats.textual_file_header = textual_file_header
    stream.stats.binary_file_header = binary_file_header
    # Also set the data encoding, endianness and the encoding of the
    # textual_file_header.
    stream.stats.data_encoding = data_encoding
    stream.stats.endian = endian
    stream.stats.textual_file_header_encoding = \
        textual_file_header_encoding

    # Convert traces to ObsPy Trace objects.
    for tr in segy_object.traces:
        stream.append(tr.to_obspy_trace(
            headonly=headonly,
            unpack_trace_headers=unpack_trace_headers))

    # Initialize empty streams for each component
    N_stream = Stream()
    E_stream = Stream()
    Z_stream = Stream()

    # Iterate through the traces and assign them to the correct stream
    for trace in stream:
        smu = trace.stats.segy.trace_header.source_measurement_unit
        if smu == 512:
            N_stream.append(trace)
        elif smu == 768:
            E_stream.append(trace)
        elif smu == 1024:
            Z_stream.append(trace)


    return [N_stream, E_stream, Z_stream]


def seismo_segy_read_textual_header(file_path):
    # Function to read and decode the Textual Header
    # Open the SEG-Y file in binary mode
    with open(file_path, 'rb') as file:
        # Read the first 3200 bytes (Textual Header)
        textual_header = file.read(3200)
        
        # Try to decode the header using ASCII first
        try:
            decoded_header = textual_header.decode('ascii')
        except UnicodeDecodeError:
            # If ASCII decoding fails, it might be EBCDIC encoded
            decoded_header = codecs.decode(textual_header, 'ebcdic-cp-us')
        
        return decoded_header

def seismo_read_trace_headers(file_path):
    directory, filename = os.path.split(file_path)
    base_name, _ = os.path.splitext(filename)
    output_file = os.path.join(directory, f"output_segy_traces_headers_{base_name}.txt")
    trace_headers = []
    
    textual_header = read_textual_header(segy_file_path)
    binary_header = read_binary_header(segy_file_path)

    #finding ns
    ns_in_binaryheader = binary_header['Number of Samples']
    ns_in_textualheader, si_in_textualheader = read_num_samples_from_textual_header(textual_header)
    ns = ns_in_binaryheader
    if isinstance(ns_in_textualheader, int):
        ns = int((10**6/si_in_textualheader)*60) #Geospace software always make traces length = 60s and also the sampling interval is in microseconds

    # Calculate the offset to the first trace data (Textual Header + Binary Header + number of external textual headers)
    num_ext_text_headers = binary_header['Number of Extended Textual File Header Records']
    offset = 3200 + 400 + (num_ext_text_headers*3200)
    
    with open(file_path, 'rb') as file:
        # Seek to the start of the first trace
        file.seek(offset)
        
        while True:
            # Read the 240-byte trace header
            trace_header = file.read(240)
            if len(trace_header) < 240:
                break  # End of file or corrupt trace
            
            # Parse number of samples from trace header (bytes 115-116)
            trace_seq_num_line = struct.unpack('>i', trace_header[0:4])[0]  # 1-4: Trace sequence number within line
            trace_seq_num_file = struct.unpack('>i', trace_header[4:8])[0]  # 5-8: Trace sequence number within file
            field_record_num = struct.unpack('>i', trace_header[8:12])[0]  # 9-12: Original field record number
            trace_num_within_field = struct.unpack('>i', trace_header[12:16])[0]  # 13-16: Trace number within field
            energy_source_point_num = struct.unpack('>i', trace_header[16:20])[0]  # 17-20: Energy source point number
            ensemble_num = struct.unpack('>i', trace_header[20:24])[0]  # 21-24: Ensemble number
            trace_num_within_ensemble = struct.unpack('>i', trace_header[24:28])[0]  # 25-28: Trace number within ensemble
            trace_id_code = struct.unpack('>h', trace_header[28:30])[0]  # 29-30: Trace identification code
            vert_sum_traces = struct.unpack('>h', trace_header[30:32])[0]  # 31-32: Number of vertically summed traces
            horiz_sum_traces = struct.unpack('>h', trace_header[32:34])[0]  # 33-34: Number of horizontally stacked traces
            data_use = struct.unpack('>h', trace_header[34:36])[0]  # 35-36: Data use
            source_receiver_distance = struct.unpack('>i', trace_header[36:40])[0]  # 37-40: Source to receiver distance
            receiver_group_elevation = struct.unpack('>i', trace_header[40:44])[0]  # 41-44: Receiver group elevation
            surface_elevation_source = struct.unpack('>i', trace_header[44:48])[0]  # 45-48: Surface elevation at source
            source_depth_below_surface = struct.unpack('>i', trace_header[48:52])[0]  # 49-52: Source depth below surface
            datum_elevation_receiver_group = struct.unpack('>i', trace_header[52:56])[0]  # 53-56: Datum elevation at receiver group
            datum_elevation_source = struct.unpack('>i', trace_header[56:60])[0]  # 57-60: Datum elevation at source
            water_depth_source = struct.unpack('>i', trace_header[60:64])[0]  # 61-64: Water depth at source
            water_depth_group = struct.unpack('>i', trace_header[64:68])[0]  # 65-68: Water depth at group
            scalar_elevations = struct.unpack('>h', trace_header[68:70])[0]  # 69-70: Scalar for elevations and depths
            scalar_coordinates = struct.unpack('>h', trace_header[70:72])[0]  # 71-72: Scalar for coordinates
            source_coordinate_x = struct.unpack('>i', trace_header[72:76])[0]  # 73-76: Source coordinate - X
            source_coordinate_y = struct.unpack('>i', trace_header[76:80])[0]  # 77-80: Source coordinate - Y
            group_coordinate_x = struct.unpack('>i', trace_header[80:84])[0]  # 81-84: Group coordinate - X
            group_coordinate_y = struct.unpack('>i', trace_header[84:88])[0]  # 85-88: Group coordinate - Y
            coordinate_units = struct.unpack('>h', trace_header[88:90])[0]  # 89-90: Coordinate units
            weathering_velocity = struct.unpack('>h', trace_header[90:92])[0]  # 91-92: Weathering velocity
            subweathering_velocity = struct.unpack('>h', trace_header[92:94])[0]  # 93-94: Subweathering velocity
            uphole_time_source_ms = struct.unpack('>h', trace_header[94:96])[0]  # 95-96: Uphole time at source
            uphole_time_group_ms = struct.unpack('>h', trace_header[96:98])[0]  # 97-98: Uphole time at group
            source_static_corr_ms = struct.unpack('>h', trace_header[98:100])[0]  # 99-100: Source static correction in milliseconds
            group_static_corr_ms = struct.unpack('>h', trace_header[100:102])[0]  # 101-102: Group static correction in milliseconds
            total_static_ms = struct.unpack('>h', trace_header[102:104])[0]  # 103-104: Total static applied in milliseconds
            lag_time_A_ms = struct.unpack('>h', trace_header[104:106])[0]  # 105-106: Lag time A in milliseconds
            lag_time_B_ms = struct.unpack('>h', trace_header[106:108])[0]  # 107-108: Lag time B in milliseconds
            delay_recording_time_ms = struct.unpack('>h', trace_header[108:110])[0]  # 109-110: Delay recording time in milliseconds
            mute_time_start_ms = struct.unpack('>h', trace_header[110:112])[0]  # 111-112: Mute time start in milliseconds
            mute_time_end_ms = struct.unpack('>h', trace_header[112:114])[0]  # 113-114: Mute time end in milliseconds
            num_samples = struct.unpack('>h', trace_header[114:116])[0]  # 115-116: Number of samples in this trace
            sample_interval = struct.unpack('>h', trace_header[116:118])[0]  # 117-118: Sample interval in microseconds
            gain_type = struct.unpack('>h', trace_header[118:120])[0]  # 119-120: Gain type of field instruments
            instrument_gain_const = struct.unpack('>h', trace_header[120:122])[0]  # 121-122: Instrument gain constant (dB)
            instrument_early_gain = struct.unpack('>h', trace_header[122:124])[0]  # 123-124: Instrument early or initial gain (dB)
            correlated = struct.unpack('>h', trace_header[124:126])[0]  # 125-126: Correlated: 1 = no, 2 = yes
            sweep_freq_start = struct.unpack('>h', trace_header[126:128])[0]  # 127-128: Sweep frequency at start (Hz)
            sweep_freq_end = struct.unpack('>h', trace_header[128:130])[0]  # 129-130: Sweep frequency at end (Hz)
            sweep_length = struct.unpack('>h', trace_header[130:132])[0]  # 131-132: Sweep length in milliseconds
            sweep_type = struct.unpack('>h', trace_header[132:134])[0]  # 133-134: Sweep type
            sweep_taper_length_start = struct.unpack('>h', trace_header[134:136])[0]  # 135-136: Sweep taper length at start in milliseconds
            sweep_taper_length_end = struct.unpack('>h', trace_header[136:138])[0]  # 137-138: Sweep taper length at end in milliseconds
            taper_type = struct.unpack('>h', trace_header[138:140])[0]  # 139-140: Taper type
            alias_filter_freq = struct.unpack('>h', trace_header[140:142])[0]  # 141-142: Alias filter frequency (Hz)
            alias_filter_slope = struct.unpack('>h', trace_header[142:144])[0]  # 143-144: Alias filter slope (dB/octave)
            notch_filter_freq = struct.unpack('>h', trace_header[144:146])[0]  # 145-146: Notch filter frequency (Hz)
            notch_filter_slope = struct.unpack('>h', trace_header[146:148])[0]  # 147-148: Notch filter slope (dB/octave)
            low_cut_freq = struct.unpack('>h', trace_header[148:150])[0]  # 149-150: Low-cut frequency (Hz)
            high_cut_freq = struct.unpack('>h', trace_header[150:152])[0]  # 151-152: High-cut frequency (Hz)
            low_cut_slope = struct.unpack('>h', trace_header[152:154])[0]  # 153-154: Low-cut slope (dB/octave)
            high_cut_slope = struct.unpack('>h', trace_header[154:156])[0]  # 155-156: High-cut slope (dB/octave)
            year_recorded = struct.unpack('>h', trace_header[156:158])[0]  # 157-158: Year data recorded
            day_of_year = struct.unpack('>h', trace_header[158:160])[0]  # 159-160: Day of year (Julian day)
            hour_of_day = struct.unpack('>h', trace_header[160:162])[0]  # 161-162: Hour of day (24-hour clock)
            minute_of_hour = struct.unpack('>h', trace_header[162:164])[0]  # 163-164: Minute of hour
            second_of_minute = struct.unpack('>h', trace_header[164:166])[0]  # 165-166: Second of minute
            time_basis_code = struct.unpack('>h', trace_header[166:168])[0]  # 167-168: Time basis code
            trace_weighting_factor = struct.unpack('>h', trace_header[168:170])[0]  # 169-170: Trace weighting factor
            geophone_roll_pos_num = struct.unpack('>h', trace_header[170:172])[0]  # 171-172: Geophone group number of roll switch position
            geophone_trace_one_num = struct.unpack('>h', trace_header[172:174])[0]  # 173-174: Geophone group number of trace number one
            geophone_last_trace_num = struct.unpack('>h', trace_header[174:176])[0]  # 175-176: Geophone group number of last trace
            gap_size = struct.unpack('>h', trace_header[176:178])[0]  # 177-178: Gap size (total number of groups dropped)
            over_travel_taper = struct.unpack('>h', trace_header[178:180])[0]  # 179-180: Over travel associated with taper at end of line
            x_coord_ensemble = struct.unpack('>i', trace_header[180:184])[0]  # 181-184: X coordinate of ensemble (CDP)
            y_coord_ensemble = struct.unpack('>i', trace_header[184:188])[0]  # 185-188: Y coordinate of ensemble (CDP)
            inline_number = struct.unpack('>i', trace_header[188:192])[0]  # 189-192: In-line number for 3-D poststack data
            crossline_number = struct.unpack('>i', trace_header[192:196])[0]  # 193-196: Cross-line number for 3-D poststack data
            shotpoint_number = struct.unpack('>i', trace_header[196:200])[0]  # 197-200: Shotpoint number
            scalar_shotpoint_number = struct.unpack('>h', trace_header[200:202])[0]  # 201-202: Scalar to be applied to the shotpoint number
            trace_value_unit = struct.unpack('>h', trace_header[202:204])[0]  # 203-204: Trace value measurement unit
            transduction_constant_mantissa = struct.unpack('>i', trace_header[204:208])[0]  # 205-208: Transduction constant mantissa
            transduction_constant_exponent = struct.unpack('>h', trace_header[208:210])[0]  # 209-210: Transduction constant exponent
            transduction_units = struct.unpack('>h', trace_header[210:212])[0]  # 211-212: Transduction units
            device_trace_identifier = struct.unpack('>h', trace_header[212:214])[0]  # 213-214: Device/Trace Identifier
            scalar_time_value = struct.unpack('>h', trace_header[214:216])[0]  # 215-216: Scalar applied to time values
            source_type_orientation = struct.unpack('>h', trace_header[216:218])[0]  # 217-218: Source type/orientation
            source_energy_direction_mantissa = struct.unpack('>i', trace_header[218:222])[0]  # 219-222: 
            source_energy_direction_exponent = struct.unpack('>h', trace_header[222:224])[0]  # 223-224: 
            source_measurement_mantissa = struct.unpack('>i', trace_header[224:228])[0]  # 225-228: Source measurement mantissa
            source_measurement_exponent = struct.unpack('>h', trace_header[228:230])[0]  # 229-230: Source measurement exponent
            source_measurement_unit = struct.unpack('>h', trace_header[230:232])[0]  # 231-232: Source measurement unit
            
            trace_headers.append({
                'Number of Samples': num_samples,
                'Trace Seq Num Line': trace_seq_num_line,
                'Trace Seq Num File': trace_seq_num_file,
                'Field Record Num': field_record_num,
                'Trace Num Within Field': trace_num_within_field,
                'Energy Source Point Num': energy_source_point_num,
                'Ensemble Num': ensemble_num,
                'Trace Num Within Ensemble': trace_num_within_ensemble,
                'Trace ID Code': trace_id_code,
                'Vert Sum Traces': vert_sum_traces,
                'Horiz Sum Traces': horiz_sum_traces,
                'Data Use': data_use,
                'Source Receiver Distance': source_receiver_distance,
                'Receiver Group Elevation': receiver_group_elevation,
                'Surface Elevation Source': surface_elevation_source,
                'Source Depth Below Surface': source_depth_below_surface,
                'Datum Elevation Receiver Group': datum_elevation_receiver_group,
                'Datum Elevation Source': datum_elevation_source,
                'Water Depth Source': water_depth_source,
                'Water Depth Group': water_depth_group,
                'Scalar Elevations': scalar_elevations,
                'Scalar Coordinates': scalar_coordinates,
                'Source Coordinate X': source_coordinate_x,
                'Source Coordinate Y': source_coordinate_y,
                'Group Coordinate X': group_coordinate_x,
                'Group Coordinate Y': group_coordinate_y,
                'Coordinate Units': coordinate_units,
                'Weathering Velocity': weathering_velocity,
                'Subweathering Velocity': subweathering_velocity,
                'Uphole Time Source (milisecond)': uphole_time_source_ms,
                'Uphole Time Group (milisecond)': uphole_time_group_ms,
                'Source Static Correction (milisecond)': source_static_corr_ms,
                'Group Static Correction (milisecond)': group_static_corr_ms,
                'Total Static Applied (milisecond)': total_static_ms,
                'Lag Time A (milisecond)': lag_time_A_ms,
                'Lag Time B (milisecond)': lag_time_B_ms,
                'Delay Recording Time (milisecond)': delay_recording_time_ms,
                'Mute Time Start (milisecond)': mute_time_start_ms,
                'Mute Time End (milisecond)': mute_time_end_ms,
                'Number of Samples': num_samples,
                'Sample Interval': sample_interval,
                'Gain Type': gain_type,
                'Instrument Gain Constant': instrument_gain_const,
                'Instrument Early Gain': instrument_early_gain,
                'Correlated': correlated,
                'Sweep Frequency Start': sweep_freq_start,
                'Sweep Frequency End': sweep_freq_end,
                'Sweep Length': sweep_length,
                'Sweep Type': sweep_type,
                'Sweep Taper Length Start': sweep_taper_length_start,
                'Sweep Taper Length End': sweep_taper_length_end,
                'Taper Type': taper_type,
                'Alias Filter Frequency': alias_filter_freq,
                'Alias Filter Slope': alias_filter_slope,
                'Notch Filter Frequency': notch_filter_freq,
                'Notch Filter Slope': notch_filter_slope,
                'Low-Cut Frequency': low_cut_freq,
                'High-Cut Frequency': high_cut_freq,
                'Low-Cut Slope': low_cut_slope,
                'High-Cut Slope': high_cut_slope,
                'Year Recorded': year_recorded,
                'Day of Year': day_of_year,
                'Hour of Day': hour_of_day,
                'Minute of Hour': minute_of_hour,
                'Second of Minute': second_of_minute,
                'Time Basis Code': time_basis_code,
                'Trace Weighting Factor': trace_weighting_factor,
                'Geophone Roll Position Number': geophone_roll_pos_num,
                'Geophone Trace One Number': geophone_trace_one_num,
                'Geophone Last Trace Number': geophone_last_trace_num,
                'Gap Size': gap_size,
                'Over Travel Taper': over_travel_taper,
                'X Coordinate Ensemble': x_coord_ensemble,
                'Y Coordinate Ensemble': y_coord_ensemble,
                'Inline Number': inline_number,
                'Crossline Number': crossline_number,
                'Shotpoint Number': shotpoint_number,
                'Scalar Shotpoint Number': scalar_shotpoint_number,
                'Trace Value Unit': trace_value_unit,
                'Transduction Constant Mantissa': transduction_constant_mantissa,
                'Transduction Constant Exponent': transduction_constant_exponent,
                'Transduction Units': transduction_units,
                'Device/Trace Identifier': device_trace_identifier,
                'Scalar Time Value': scalar_time_value,
                'Source Type/Orientation': source_type_orientation,
                'Source Energy Direction Mentissa': source_energy_direction_mantissa,
                'Source Energy Direction Exponent': source_energy_direction_exponent,
                'Source Measurement Mantissa': source_measurement_mantissa,
                'Source Measurement Exponent': source_measurement_exponent,
                'Source Measurement Unit': source_measurement_unit,
                # Add other fields here...
            })

            # Read the trace data
            trace_data = file.read(ns * 4)  # Assuming 4 bytes per sample (Data Sample Format = 5)
            
            if len(trace_data) < ns * 4:
                break  # End of file or corrupt trace data
            
    with open(output_file, 'w') as file:
        for i, trace in enumerate(trace_headers, 1):
            for key, value in trace.items():
                file.write(f"{key}: {value}, ")
            file.write("}\n")  # Close the dictionary and move to the next line for the next trace


def seismo_get_segy_number_of_traces(file_path):
    with open(file_path, 'rb') as file:
        file.seek(3200)
        binary_header = file.read(400)
        num_extended_headers = struct.unpack('>h', binary_header[304:306])[0]
        dt = struct.unpack('>h', binary_header[16:18])[0]
        ns = int((10**6/dt)*60)
        start = file.seek(3600 + (num_extended_headers*3200))
        file_size = os.path.getsize(file_path)
        ntraces = (file_size-start) / (240+(ns*4))
    return ntraces


def seismo_segy_remove_extended_headers(file_path):
    """

    Warning: This function is written for reading the SEGY files coming from the Geospace GSB3s 
    at Institute of seismology - University of Helsinki. SEGY files with other configurations might
    not work 

    For the segy files that have extended headers
    Obspy do not read Segy files with exteneded headers. The extended headers 
    can be removed and the parameter in the binary header pointing out to the extended 
    headers should change to 0. 

    This function Reads a SEG Y file and returns another SEG Y file with no extended header
    The output will be saved in the same directory as the input path with "removed_ex_headers_" followed by 
    the input name

    :type file_path: String
    :param file_path: local path of the SEG Y rev1 file to be read.

    :Returns SEG Y File: The same SEG Y file without Extended Header
    """
    input_dir = os.path.dirname(file_path)
    input_name = os.path.basename(file_path)
    output_name = "removed_ex_headers_" + input_name
    output_segy = os.path.join(input_dir, output_name)
    with open(file_path, 'rb') as infile:
        # Step 1: Read the initial 3200-byte textual header
        textual_header = infile.read(3200)

        # Step 2: Read the 400-byte binary header
        binary_header = bytearray(infile.read(400))  # Use bytearray to allow modifications

        # Step 3: Extract the number of extended textual headers from the binary header
        num_extended_headers = struct.unpack('>h', binary_header[304:306])[0]
        print(f"Number of Extended Textual Headers: {num_extended_headers}")
        
        if(num_extended_headers > 0):
            # Modify the bytes 305-306 in the binary header to set the number of extended headers to 0
            binary_header[304:306] = struct.pack('>h', 0)

            # Step 4: Skip the extended textual headers
            infile.seek(num_extended_headers * 3200, os.SEEK_CUR)

            # Step 5: Read the rest of the file and write to a new file
            with open(output_segy, 'wb') as outfile:
                # Write the initial 3200-byte textual header
                outfile.write(textual_header)

                # Write the modified 400-byte binary header
                outfile.write(binary_header)

                # Write the rest of the data (starting from the first trace header)
                while True:
                    data = infile.read(4096)
                    if not data:
                        break
                    outfile.write(data)
            return outfile
        elif:
            print("No extended header were identified, No changes were made. aborting...")
    return 


def seismo_segy_read_extended_textual_headers(file_path):
    """
    This function reads the extended textual headers (if there is any)

    :type file_path: String
    :param file_path: local path of the SEG Y rev1 file to be read.

    :Returns String: Extended Textual Headers
    """
    headers = []
    bytesize = 0
    with open(file_path, 'rb') as file:
        # Skip the first 3600 bytes (Textual Header + Binary Header)
        file.seek(3600)
        
        while True:
            # Read the next 3200 bytes to check for the Extended Textual File Header
            extended_header = file.read(3200)
            
            # If the block is empty, we've reached the end of the extended headers
            if not extended_header:
                break
            
            # Check if it's a valid text block (Extended Textual Header)
            if all(32 <= byte <= 126 or byte == 10 or byte == 13 for byte in extended_header):
                headers.append(extended_header.decode('ascii').strip())
                bytesize+=len(extended_header)
            else:
                break  # Exit the loop if the block is not a textual header
    
    formatted_header = ""
    for header in headers:
        formatted_header += header.replace("\r\n", "\n") + "\n"
    return formatted_header


def seismo_segy_read_num_samples_from_textual_header(textual_header):
    # Define regular expression patterns to find the relevant information
    sample_interval_pattern = re.compile(r'SAMPLE INTERVAL:\s*([\d.]+)\s*msec')
    num_samples_pattern = re.compile(r'SAMPLES/TRACE:\s*(\d+)')

    # Search the header for the patterns
    ns_text = num_samples_pattern.search(textual_header)
    sam_interval = sample_interval_pattern.search(textual_header)

    # Extract the number of samples from the match object, if found
    if ns_text:
        ns = int(ns_text.group(1))
    else:
        ns = None

    # Extract the sample interval from the match object, if found, and convert to microseconds
    if sam_interval:
        # Convert sample interval from milliseconds to microseconds
        si_ms = float(sam_interval.group(1))
        si = si_ms * 1000
    else:
        si = None

    return (ns, si)


def seismo_segy_read_binary_header(file_path):
    with open(file_path, 'rb') as file:
        # Skip the Textual Header (first 3200 bytes)
        file.seek(3200)
        # Read the next 400 bytes (Binary Header)
        binary_header = file.read(400)
        # Parse important fields from the Binary Header using struct.unpack
        # Job Identification Number (bytes 1-4)
        job_id = struct.unpack('>i', binary_header[0:4])[0]
        # Line Number (bytes 5-8)
        line_number = struct.unpack('>i', binary_header[4:8])[0]
        # Reel Number (bytes 9-12)
        reel_number = struct.unpack('>i', binary_header[8:12])[0]
        # Number of Traces per Ensemble (bytes 13-14)
        traces_per_ensemble = struct.unpack('>h', binary_header[12:14])[0]
        # Number of Traces per Ensemble (bytes 15-16)
        auxiliary_trace_per_ensemble  = struct.unpack('>h', binary_header[14:16])[0]
        # Sample interval in microseconds (bytes 17-18)
        dt = struct.unpack('>h', binary_header[16:18])[0]
        # Sample interval in microseconds of original field recording (bytes 19-20)
        dtOrig = struct.unpack('>h', binary_header[18:20])[0]
        # Number of Samples (bytes 21-22)
        ns = struct.unpack('>h', binary_header[20:22])[0]
        # Number of Samples per data trace for original field recording (bytes 23-24)
        nsOrig = struct.unpack('>h', binary_header[22:24])[0]
        # Data Sample Format (bytes 25-26)
        data_sample_format = struct.unpack('>h', binary_header[24:26])[0]
        # Ensemble fold (bytes 27-28))
        ensemble_fold = struct.unpack('>h', binary_header[26:28])[0]
        # Trace Sorting (bytes 29-30))
        trace_sorting = struct.unpack('>h', binary_header[28:30])[0]
        # Vertical Sum Code (bytes 31-32)
        vertical_sum_code = struct.unpack('>h', binary_header[30:32])[0]
        # Sweep Frequency at Start (bytes 33-34)
        sweep_freq_start = struct.unpack('>h', binary_header[32:34])[0]
        # Sweep Frequency at End (bytes 35-36)
        sweep_freq_end = struct.unpack('>h', binary_header[34:36])[0]
        # Sweep Length in ms (bytes 37-38)
        sweep_length = struct.unpack('>h', binary_header[36:38])[0]
        # Sweep Type Code (bytes 39-40)
        sweep_type = struct.unpack('>h', binary_header[38:40])[0]
        # Trace Number of Sweep Channel (bytes 41-42)
        trace_sweep_channel = struct.unpack('>h', binary_header[40:42])[0]
        # Sweep Trace Taper Length at Start (bytes 43-44)
        sweep_taper_start = struct.unpack('>h', binary_header[42:44])[0]
        # Sweep Trace Taper Length at End (bytes 45-46)
        sweep_taper_end = struct.unpack('>h', binary_header[44:46])[0]
        # Taper Type (bytes 47-48)
        taper_type = struct.unpack('>h', binary_header[46:48])[0]
        # Correlated Data Traces (bytes 49-50)
        correlated_traces = struct.unpack('>h', binary_header[48:50])[0]
        # Binary Gain Recovered (bytes 51-52)
        binary_gain = struct.unpack('>h', binary_header[50:52])[0]
        # Amplitude Recovery Method (bytes 53-54)
        amplitude_recovery = struct.unpack('>h', binary_header[52:54])[0]
        # Measurement System (bytes 55-56)
        measurement_system = struct.unpack('>h', binary_header[54:56])[0]
        # Impulse Signal Polarity (bytes 57-58)
        signal_polarity = struct.unpack('>h', binary_header[56:58])[0]
        # Vibratory Polarity Code (bytes 59-60)
        vibratory_polarity = struct.unpack('>h', binary_header[58:60])[0]
        # SEG-Y Format Revision Number (bytes 301-302)
        seg_y_revision = struct.unpack('>H', binary_header[300:302])[0]
        # Fixed Length Trace Flag (bytes 303-304)
        fixed_length_trace_flag = struct.unpack('>h', binary_header[302:304])[0]
        # Number of Extended Textual File Header Records (bytes 305-306)
        num_extended_headers = struct.unpack('>h', binary_header[304:306])[0]
        # Unassigned (bytes 307-400)
        unassigned = struct.unpack('>47h', binary_header[306:400])

        return {
            'Job ID': job_id,
            'Line Number': line_number,
            'Reel Number': reel_number,
            'Traces Per Ensemble': traces_per_ensemble,
            'Auxiliary Traces Per Ensemble': auxiliary_trace_per_ensemble,
            'Sampling Interval': dt,
            'Sampling Interval Original': dtOrig,
            'Number of Samples': ns,
            'Number of Samples Original': nsOrig,
            'Data Sample Format': data_sample_format,
            'Ensemble Fold': ensemble_fold,
            'Trace Sorting': trace_sorting,
            'Vertical Sum Code': vertical_sum_code,
            'Sweep Frequency Start': sweep_freq_start,
            'Sweep Frequency End': sweep_freq_end,
            'Sweep Length': sweep_length,
            'Sweep Type': sweep_type,
            'Trace Number of Sweep Channel': trace_sweep_channel,
            'Sweep Taper Start Length': sweep_taper_start,
            'Sweep Taper End Length': sweep_taper_end,
            'Taper Type': taper_type,
            'Correlated Data Traces': correlated_traces,
            'Binary Gain Recovered': binary_gain,
            'Amplitude Recovery Method': amplitude_recovery,
            'Measurement System': measurement_system,
            'Impulse Signal Polarity': signal_polarity,
            'Vibratory Polarity': vibratory_polarity,
            'SEG-Y Format Revision Number': seg_y_revision,
            'Fixed Length Trace Flag': fixed_length_trace_flag,
            'Number of Extended Textual File Header Records': num_extended_headers,
            'Unassigned': unassigned
        }


def seismo_plot_two_streams_fdomain(stream1: Stream, stream2: Stream):
    """
    It plots two streams spectrum next to each other for comparison
    """
    # Create a figure with subplots
    num_traces1 = len(stream1)
    num_traces2 = len(stream2)
    fig, axs = plt.subplots(nrows=max(num_traces1,num_traces2), ncols=2, figsize=(10, 6), sharex=True)
    plt.xlim(0, 2)
    # Plot the first Stream
    for i, trace in enumerate(stream1):
        fft_data, frequencies = fftspectrum(trace)
        axs[i, 0].plot(frequencies, np.abs(fft_data), label=trace.id)
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend()
        axs[i, 0].set_title(f"F spectrum - Channel {i + 1}")

    # Plot the second Stream
    for j, trace in enumerate(stream2):
        fft_data, frequencies = fftspectrum(trace)
        axs[j, 1].plot(frequencies, np.abs(fft_data), label=trace.id)
        axs[j, 1].set_ylabel("Amplitude")
        axs[j, 1].legend()
        axs[j, 1].set_title(f"F spectrum - Chwannel {i + 1}")

    # Set common labels
    # axs[-1, 0].set_xlabel("Time (s)")
    # axs[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def _write_segy(stream, filename, data_encoding=None, byteorder=None,
                textual_header_encoding=None, **kwargs):  # @UnusedVariable
    """
    Writes a SEG Y file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type data_encoding: int
    :param data_encoding: The data encoding is an integer with the following
        currently supported meaning:

        ``1``
            4 byte IBM floating points (float32)
        ``2``
            4 byte Integers (int32)
        ``3``
            2 byte Integer (int16)
        ``5``
            4 byte IEEE floating points (float32)

        The value in the brackets is the necessary dtype of the data. ObsPy
        will now automatically convert the data because data might change/loose
        precision during the conversion so the user has to take care of the
        correct dtype.

        If it is ``None``, the value of the first Trace will be used for all
        consecutive Traces. If it is None for the first Trace, 1 (IBM floating
        point numbers) will be used. Different data encodings for different
        traces are currently not supported because these will most likely not
        be readable by other software.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If is ``None``, it will either
        be the endianness of the first Trace or if that is also not set, it
        will be big endian. A mix between little and big endian for the headers
        and traces is currently not supported.
    :type textual_header_encoding: str or ``None``
    :param textual_header_encoding: The encoding of the textual header. Can be
        ``'EBCDIC'``, ``'ASCII'`` or ``None``. If it is ``None``, the
        textual_file_header_encoding attribute in the stats.segy dictionary of
        the first Trace is used and if that is not set, ASCII will be used.

    This function will automatically set the data encoding field of the binary
    file header so the user does not need to worry about it.

    The starttime of every trace is not a required field in the SEG Y
    specification. If the starttime of a trace is UTCDateTime(0) it will be
    interpreted as a not-set starttime and no time is written to the trace
    header. Every other time will be written.

    SEG Y supports a sample interval from 1 to 65535 microseconds in steps of 1
    microsecond. Larger intervals cannot be supported due to the definition of
    the SEG Y format. Therefore the smallest possible sampling rate is ~ 15.26
    Hz. Please keep that in mind.
    """
    for i, tr in enumerate(stream):
        if len(tr) > MAX_NUMBER_OF_SAMPLES:
            msg = ('Can not write traces with more than {:d} samples (trace '
                   'at index {:d}):\n{!s}')
            raise ValueError(msg.format(MAX_NUMBER_OF_SAMPLES, i, tr))

    # Some sanity checks to catch invalid arguments/keyword arguments.
    if data_encoding is not None and data_encoding not in VALID_FORMATS:
        msg = "Invalid data encoding."
        raise SEGYCoreWritingError(msg)
    # Figure out the data encoding if it is not set.
    if data_encoding is None:
        if hasattr(stream, 'stats') and hasattr(stream.stats, 'data_encoding'):
            data_encoding = stream.stats.data_encoding
        if hasattr(stream, 'stats') and hasattr(stream.stats,
                                                'binary_file_header'):
            data_encoding = \
                stream.stats.binary_file_header.data_sample_format_code
        # Set it to float if it in not given.
        else:
            data_encoding = 1

    # Create empty file wide headers if they do not exist.
    if not hasattr(stream, 'stats'):
        stream.stats = AttribDict()
    if not hasattr(stream.stats, 'textual_file_header'):
        stream.stats.textual_file_header = b""
    if not hasattr(stream.stats, 'binary_file_header'):
        stream.stats.binary_file_header = SEGYBinaryFileHeader()

    # Valid dtype for the data encoding.
    valid_dtype = DATA_SAMPLE_FORMAT_CODE_DTYPE[data_encoding]
    # Makes sure that the dtype is for every Trace is correct.
    for trace in stream:
        # Check the dtype.
        if trace.data.dtype != valid_dtype:
            msg = """
            The dtype of the data and the chosen data_encoding do not match.
            You need to manually convert the dtype if you want to use that
            data_encoding. Please refer to the obspy.io.segy manual for more
            details.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check the sample interval.
        if trace.stats.delta > MAX_INTERVAL_IN_SECONDS:
            msg = """
            SEG Y supports a maximum interval of %s seconds in between two
            samples (trace.stats.delta value).
            """.strip()
            msg = msg % MAX_INTERVAL_IN_SECONDS
            raise SEGYSampleIntervalError(msg)

    # Figure out endianness and the encoding of the textual file header.
    if byteorder is None:
        if hasattr(stream, 'stats') and hasattr(stream.stats, 'endian'):
            byteorder = stream.stats.endian
        else:
            byteorder = '>'
    # Map the byte order.
    byteorder = ENDIAN[byteorder]
    if textual_header_encoding is None:
        if hasattr(stream, 'stats') and hasattr(
                stream.stats, 'textual_file_header_encoding'):
            textual_header_encoding = \
                stream.stats.textual_file_header_encoding
        else:
            textual_header_encoding = 'ASCII'

    # Loop over all Traces and create a SEGY File object.
    segy_file = SEGYFile()
    # Set the file wide headers.
    segy_file.textual_file_header = stream.stats.textual_file_header
    segy_file.textual_header_encoding = \
        textual_header_encoding
    binary_header = SEGYBinaryFileHeader()
    this_binary_header = stream.stats.binary_file_header
    # Loop over all items and if they exists set them. Ignore all other
    # attributes.
    for _, item, _ in BINARY_FILE_HEADER_FORMAT:
        if hasattr(this_binary_header, item):
            setattr(binary_header, item, getattr(this_binary_header, item))
    # Set the data encoding.
    binary_header.data_sample_format_code = data_encoding
    segy_file.binary_file_header = binary_header
    # Add all traces.
    for trace in stream:
        new_trace = SEGYTrace()
        new_trace.data = trace.data
        # Create empty trace header if none is there.
        if not hasattr(trace.stats, 'segy'):
            warnings.warn("CREATING TRACE HEADER")
            trace.stats.segy = {}
            trace.stats.segy.trace_header = SEGYTraceHeader(endian=byteorder)
        elif not hasattr(trace.stats.segy, 'trace_header'):
            warnings.warn("CREATING TRACE HEADER")
            trace.stats.segy.trace_header = SEGYTraceHeader()
        this_trace_header = trace.stats.segy.trace_header
        new_trace_header = new_trace.header
        # Again loop over all field of the trace header and if they exists, set
        # them. Ignore all additional attributes.
        for _, item, _, _ in TRACE_HEADER_FORMAT:
            if hasattr(this_trace_header, item):
                setattr(new_trace_header, item,
                        getattr(this_trace_header, item))
        starttime = trace.stats.starttime
        # Set the date of the Trace if it is not UTCDateTime(0).
        if starttime == UTCDateTime(0):
            new_trace.header.year_data_recorded = 0
            new_trace.header.day_of_year = 0
            new_trace.header.hour_of_day = 0
            new_trace.header.minute_of_hour = 0
            new_trace.header.second_of_minute = 0
        else:
            new_trace.header.year_data_recorded = starttime.year
            new_trace.header.day_of_year = starttime.julday
            new_trace.header.hour_of_day = starttime.hour
            new_trace.header.minute_of_hour = starttime.minute
            new_trace.header.second_of_minute = starttime.second
        # Set the sampling rate.
        new_trace.header.sample_interval_in_ms_for_this_trace = \
            int(trace.stats.delta * 1E6)
        # Set the data encoding and the endianness.
        new_trace.data_encoding = data_encoding
        new_trace.endian = byteorder
        # Add the trace to the SEGYFile object.
        segy_file.traces.append(new_trace)
    # Write the file
    segy_file.write(filename, data_encoding=data_encoding, endian=byteorder)


def _is_su(file):
    """
    Checks whether or not the given file is a Seismic Unix (SU) file.

    :type file: str or file-like object
    :param file: Seismic Unix file to be checked.
    :rtype: bool
    :return: ``True`` if a Seismic Unix file.

    .. note::
        This test is rather shaky because there is no reliable identifier in a
        Seismic Unix file.
    """
    with open_bytes_stream(file) as f:
        stat = autodetect_endian_and_sanity_check_su(f)
    if stat is False:
        return False
    else:
        return True


def _read_su(filename, headonly=False, byteorder=None,
             unpack_trace_headers=False, **kwargs):  # @UnusedVariable
    """
    Reads a Seismic Unix (SU) file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SU file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the header and omit the waveform
        data.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If it is ``None``, it will try
        to autodetect the endianness. The endianness is always valid for the
        whole file. Defaults to ``None``.
    :type unpack_trace_headers: bool, optional
    :param unpack_trace_headers: Determines whether or not all trace header
        values will be unpacked during reading. If ``False`` it will greatly
        enhance performance and especially memory usage with large files. The
        header values can still be accessed and will be calculated on the fly
        but tab completion will no longer work. Look in the headers.py for a
        list of all possible trace header values. Defaults to ``False``.
    :returns: A ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/1.su_first_trace")
    >>> st #doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  #doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    ... | 2005-12-19T15:07:54.000000Z - ... | 4000.0 Hz, 8000 samples
    """
    # Read file to the internal segy representation.
    su_object = _read_su_file(filename, endian=byteorder,
                              unpack_headers=unpack_trace_headers)

    # Create the stream object.
    stream = Stream()

    # Get the endianness from the first trace.
    endian = su_object.traces[0].endian
    # Loop over all traces.
    for tr in su_object.traces:
        # Create new Trace object for every segy trace and append to the Stream
        # object.
        trace = Trace()
        stream.append(trace)
        # skip data if headonly is set
        if headonly:
            trace.stats.npts = tr.npts
        else:
            trace.data = tr.data
        trace.stats.su = AttribDict()
        # If all values will be unpacked create a normal dictionary.
        if unpack_trace_headers:
            # Add the trace header as a new attrib dictionary.
            header = AttribDict()
            for key, value in tr.header.__dict__.items():
                setattr(header, key, value)
        # Otherwise use the LazyTraceHeaderAttribDict.
        else:
            # Add the trace header as a new lazy attrib dictionary.
            header = LazyTraceHeaderAttribDict(tr.header.unpacked_header,
                                               tr.header.endian)
        trace.stats.su.trace_header = header
        # Also set the endianness.
        trace.stats.su.endian = endian
        # The sampling rate should be set for every trace. It is a sample
        # interval in microseconds. The only sanity check is that is should be
        # larger than 0.
        tr_header = trace.stats.su.trace_header
        if tr_header.sample_interval_in_ms_for_this_trace > 0:
            trace.stats.delta = \
                float(tr.header.sample_interval_in_ms_for_this_trace) / \
                1E6
        # If the year is not zero, calculate the start time. The end time is
        # then calculated from the start time and the sampling rate.
        # 99 is often used as a placeholder.
        if tr_header.year_data_recorded > 0:
            year = tr_header.year_data_recorded
            # The SEG Y rev 0 standard specifies the year to be a 4 digit
            # number.  Before that it was unclear if it should be a 2 or 4
            # digit number. Old or wrong software might still write 2 digit
            # years. Every number <30 will be mapped to 2000-2029 and every
            # number between 30 and 99 will be mapped to 1930-1999.
            if year < 100:
                if year < 30:
                    year += 2000
                else:
                    year += 1900
            julday = tr_header.day_of_year
            julday = tr_header.day_of_year
            hour = tr_header.hour_of_day
            minute = tr_header.minute_of_hour
            second = tr_header.second_of_minute
            trace.stats.starttime = UTCDateTime(
                year=year, julday=julday, hour=hour, minute=minute,
                second=second)
    return stream


def _write_su(stream, filename, byteorder=None, **kwargs):  # @UnusedVariable
    """
    Writes a Seismic Unix (SU) file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If is ``None``, it will either
        be the endianness of the first Trace or if that is also not set, it
        will be big endian. A mix between little and big endian for the headers
        and traces is currently not supported.

    This function will automatically set the data encoding field of the binary
    file header so the user does not need to worry about it.
    """
    # Check that the dtype for every Trace is correct.
    for trace in stream:
        # Check the dtype.
        if trace.data.dtype != np.float32:
            msg = """
            The dtype of the data is not float32.  You need to manually convert
            the dtype. Please refer to the obspy.io.segy manual for more
            details.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check the sample interval.
        if trace.stats.delta > MAX_INTERVAL_IN_SECONDS:
            msg = """
            Seismic Unix supports a maximum interval of %s seconds in between
            two samples (trace.stats.delta value).
            """.strip()
            msg = msg % MAX_INTERVAL_IN_SECONDS
            raise SEGYSampleIntervalError(msg)

    # Figure out endianness and the encoding of the textual file header.
    if byteorder is None:
        if hasattr(stream[0].stats, 'su') and hasattr(stream[0].stats.su,
                                                      'endian'):
            byteorder = stream[0].stats.su.endian
        else:
            byteorder = '>'

    # Loop over all Traces and create a SEGY File object.
    su_file = SUFile()
    # Add all traces.
    for trace in stream:
        new_trace = SEGYTrace()
        new_trace.data = trace.data
        # Use header saved in stats if one exists.
        if hasattr(trace.stats, 'su') and \
           hasattr(trace.stats.su, 'trace_header'):
            this_trace_header = trace.stats.su.trace_header
        else:
            this_trace_header = AttribDict()
        new_trace_header = new_trace.header
        # Again loop over all field of the trace header and if they exists, set
        # them. Ignore all additional attributes.
        for _, item, _, _ in TRACE_HEADER_FORMAT:
            if hasattr(this_trace_header, item):
                setattr(new_trace_header, item,
                        getattr(this_trace_header, item))
        starttime = trace.stats.starttime
        # Set some special attributes, e.g. the sample count and other stuff.
        new_trace_header.number_of_samples_in_this_trace = trace.stats.npts
        new_trace_header.sample_interval_in_ms_for_this_trace = \
            int(round((trace.stats.delta * 1E6)))
        # Set the date of the Trace if it is not UTCDateTime(0).
        if starttime == UTCDateTime(0):
            new_trace.header.year_data_recorded = 0
            new_trace.header.day_of_year = 0
            new_trace.header.hour_of_day = 0
            new_trace.header.minute_of_hour = 0
            new_trace.header.second_of_minute = 0
        else:
            new_trace.header.year_data_recorded = starttime.year
            new_trace.header.day_of_year = starttime.julday
            new_trace.header.hour_of_day = starttime.hour
            new_trace.header.minute_of_hour = starttime.minute
            new_trace.header.second_of_minute = starttime.second
        # Set the data encoding and the endianness.
        new_trace.endian = byteorder
        # Add the trace to the SEGYFile object.
        su_file.traces.append(new_trace)
    # Write the file
    su_file.write(filename, endian=byteorder)


def _segy_trace_str_(self, *args, **kwargs):
    """
    Monkey patch for the __str__ method of the Trace object. SEGY object do not
    have network, station, channel codes. It just prints the trace sequence
    number within the line.
    """
    try:
        out = "%s" % (
            'Seq. No. in line: %4i' %
            self.stats.segy.trace_header.trace_sequence_number_within_line)
    except (KeyError, AttributeError):
        # fall back if for some reason the segy attribute does not exists
        return getattr(Trace, '__original_str__')(self, *args, **kwargs)
    # output depending on delta or sampling rate bigger than one
    if self.stats.sampling_rate < 0.1:
        if hasattr(self.stats, 'preview') and self.stats.preview:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(delta).1f s, %(npts)d samples [preview]"
        else:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(delta).1f s, %(npts)d samples"
    else:
        if hasattr(self.stats, 'preview') and self.stats.preview:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(sampling_rate).1f Hz, %(npts)d samples [preview]"
        else:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(sampling_rate).1f Hz, %(npts)d samples"
    # check for masked array
    if np.ma.count_masked(self.data):
        out += ' (masked)'
    return out % (self.stats)


class LazyTraceHeaderAttribDict(AttribDict):
    """
    This version of AttribDict will unpack header values only if needed.

    This saves a huge amount of memory. The disadvantage is that it is no
    longer possible to use tab completion in e.g. ipython.

    This version is used for the SEGY/SU trace headers.
    """
    readonly = ["unpacked_header", "endian"]

    def __init__(self, unpacked_header, unpacked_header_endian, data={}):
        dict.__init__(data)
        self.update(data)
        self.__dict__['unpacked_header'] = unpacked_header
        self.__dict__['endian'] = unpacked_header_endian

    def __getitem__(self, name):
        # Return if already set.
        if name in self.__dict__:
            return self.__dict__[name]
        # Otherwise try to unpack them.
        try:
            index = TRACE_HEADER_KEYS.index(name)
        # If not found raise an attribute error.
        except ValueError:
            msg = "'%s' object has no attribute '%s'" % \
                (self.__class__.__name__, name)
            raise AttributeError(msg)
        # Unpack the one value and set the class attribute so it will does not
        # have to unpacked again if accessed in the future.
        length, name, special_format, start = TRACE_HEADER_FORMAT[index]
        string = self.__dict__['unpacked_header'][start: start + length]
        attribute = unpack_header_value(self.__dict__['endian'], string,
                                        length, special_format)
        setattr(self, name, attribute)
        return attribute

    __getattr__ = __getitem__

    def __deepcopy__(self, *args, **kwargs):  # @UnusedVariable, see #689
        ad = self.__class__(
            unpacked_header=deepcopy(self.__dict__['unpacked_header']),
            unpacked_header_endian=deepcopy(self.__dict__['endian']),
            data=dict((k, deepcopy(v)) for k, v in self.__dict__.items()
                      if k not in self.readonly))
        return ad


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)


# Monkey patch the __str__ method for the all Trace instances used in the
# following.
# XXX: Check if this is not messing anything up. Patching every single
# instance did not reliably work.
setattr(Trace, '__original_str__', Trace.__str__)
setattr(Trace, '__str__', _segy_trace_str_)


