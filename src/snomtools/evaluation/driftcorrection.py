"""
This file provides driftkorrection for array stacks. It generates the drift vectors via
Crosscorrelation-Methods provided by the OpenCV library

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import cv2 as cv
import numpy as np
import snomtools.data.datasets
import snomtools.data.h5tools
import snomtools.data.transformation.rotate as rot
from snomtools.data.tools import iterfy, full_slice, sliced_shape

__author__ = 'Benjamin Frisch'

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False


class Drift(object):
    # TODO: Implement usage of more than one DataArray in the DataSet.

    def __init__(self, data=None, precalculated_drift=None, template=None, stackAxisID=None, yAxisID=None, xAxisID=None,
                 subpixel=True, method='cv.TM_CCOEFF_NORMED', template_origin=None, interpolation_order=None):
        """
        Calculates the correlation of a given 2D template with all slices in a n-D dataset which gets projected onto the
        three axes stackAxis, yAxis, xAxis.
        Different methods and subpixel accuracy are available.

        :param data: n-D Dataset to be driftcorrected.
        :type data: snomtools.data.datasets.Dataset **or** snomtools.data.datasets.ROI

        :param template: A template dataset to match to. This is typically a subset of ROI of :code:`data`.
        :type template: snomtools.data.datasets.Dataset **or** snomtools.data.datasets.ROI

        :param stackAxisID: Axis, along which the template_matching is calculated

        :param yAxisID: ID of the first axis of the image, i.e. y

        :param xAxisID: ID of the second axis of the image, i.e. x

        :param subpixel: Generate subpixel accurate drift vectors

        :param method: Method to calculate the Correlation between template and data. Possible methods:
            'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
            'cv.TM_SQDIFF_NORMED'

        :param template_origin: An origin for the relative drift vectors in the form (y_pixel, x_pixel). If :code:`None`
            is given (the default), the first detected drift vector along stackAxis is used.
        :type template_origin: tuple(int **or** float) of len==2 **or** None

        :param int interpolation_order: An order for the interpolation for the calculation of driftcorrected data.
            See: :func:`scipy.ndimage.interpolation.shift` for details.
        """
        if data is None:
            if precalculated_drift is None:
                self.drift = None
            else:
                assert len(precalculated_drift[0]) == 2, "Driftvector has not dimension 2"
                self.drift = precalculated_drift

        else:
            if stackAxisID is None:
                self.dstackAxisID = data.get_axis_index('delay')
            else:
                if type(stackAxisID) is int:
                    self.dstackAxisID = stackAxisID
                else:
                    self.dstackAxisID = data.get_axis_index(stackAxisID)
            if yAxisID is None:
                self.dyAxisID = data.get_axis_index('y')
            else:
                if type(yAxisID) is int:
                    self.dyAxisID = yAxisID
                else:
                    self.dyAxisID = data.get_axis_index(yAxisID)
            if xAxisID is None:
                self.dxAxisID = data.get_axis_index('x')
            else:
                if type(xAxisID) is int:
                    self.dxAxisID = xAxisID
                else:
                    self.dxAxisID = data.get_axis_index(xAxisID)

            # read or guess template
            if template is None:
                self.template = self.guess_templatedata(data, self.dyAxisID, self.dxAxisID)

            else:
                if yAxisID is None:
                    tyAxisID = template.get_axis_index('y')
                else:
                    if type(yAxisID) is int:
                        tyAxisID = yAxisID
                    else:
                        tyAxisID = template.get_axis_index(yAxisID)
                if xAxisID is None:
                    txAxisID = template.get_axis_index('x')
                else:
                    if type(xAxisID) is int:
                        txAxisID = xAxisID
                    else:
                        txAxisID = template.get_axis_index(xAxisID)
                self.template = self.extract_templatedata(template, tyAxisID, txAxisID)

            # check for external drift vectors
            if precalculated_drift is None:
                # process data towards 3d array
                if verbose:
                    print("Projecting 3D data...", end=None)
                self.data3D = self.extract_3Ddata(data, self.dstackAxisID, self.dyAxisID, self.dxAxisID)
                if verbose:
                    print("...done")

                # for layers along stackAxisID find drift:
                self.drift = self.template_matching_stack(self.data3D.get_datafield(0), self.template,
                                                          self.dstackAxisID,
                                                          method=method, subpixel=subpixel)

            else:
                assert len(precalculated_drift) == data.shape[
                    self.dstackAxisID], "Number of driftvectors unequal to stack dimension of data"
                assert len(precalculated_drift[0]) == 2, "Driftvector has not dimension 2"
                self.drift = precalculated_drift

        if template_origin is None:
            if self.drift is not None:
                self.template_origin = self.drift[0]
            else:
                self.template_origin = None
        else:
            template_origin = tuple(template_origin)
            assert len(template_origin) == 2, "template_origin has invalid length."
            self.template_origin = template_origin

        self.data = data
        self.subpixel = subpixel
        if interpolation_order is None:
            if self.subpixel:
                self.interpolation_order = 1
            else:
                self.interpolation_order = 0
        else:
            self.interpolation_order = interpolation_order

    @property
    def drift_relative(self):
        """
        The relative drift vector list, as in difference to the template_origin.

        :return: List of (y-difference, x-difference) given in Pixels (Array Indices).
        """
        return self.as_relative_vectors(self.drift)

    @property
    def xdrift(self):
        """
        The x-components of the drift vector list.

        :return: List of x-values given in Pixels (Array Indices).
        """
        return np.array([t[1] for t in self.drift])

    @property
    def xdrift_relative(self):
        """
        The x-components of the relative drift vector list (Relative to template_origin).

        :return: List of x-difference-values given in Pixels (Array Indices).
        """
        return np.array([t[1] for t in self.drift_relative])

    @property
    def ydrift(self):
        """
        The y-components of the drift vector list.

        :return: List of y-values given in Pixels (Array Indices).
        """
        return np.array([t[0] for t in self.drift])

    @property
    def ydrift_relative(self):
        """
        The y-components of the relative drift vector list (Relative to template_origin).

        :return: List of y-difference-values given in Pixels (Array Indices).
        """
        return np.array([t[0] for t in self.drift_relative])

    @property
    def driftdata(self):
        """
        The detected drift data collected in a DataSet. The DataSet will be 1-D along the drift axis and will contain
        multiple DataArrays, representing the absolute and relative drift components along both image dimensions.
        Also, the corresponding (absolute and relative) position on the x and y Axis will be calculated
        (E.g. how many micrometers (not pixels) did the image drift in the x and y direction.).

        :return: A DataSet containing the assembled drift data.
        """
        d_axis = self.data.get_axis(self.dstackAxisID)
        y_axis = self.data.get_axis(self.dyAxisID)
        x_axis = self.data.get_axis(self.dxAxisID)
        das = []
        das.append(snomtools.data.datasets.DataArray(self.ydrift, label="Axis {0} drift index".format(y_axis.label)))
        das.append(snomtools.data.datasets.DataArray(self.ydrift_relative,
                                                     label="Axis {0} drift relative".format(y_axis.label)))
        das.append(snomtools.data.datasets.DataArray([y_axis.value_floatindex(y).magnitude for y in self.ydrift],
                                                     unit=y_axis.units,
                                                     label="Axis {0} drift position".format(y_axis.label)))
        das.append(snomtools.data.datasets.DataArray(
            [(y_axis.value_floatindex(y) - y_axis.value_floatindex(self.template_origin[0])).magnitude
             for y in self.ydrift],
            unit=y_axis.units,
            label="Axis {0} drift relative position".format(y_axis.label)))

        das.append(snomtools.data.datasets.DataArray(self.xdrift, label="Axis {0} drift index".format(x_axis.label)))
        das.append(snomtools.data.datasets.DataArray(self.xdrift_relative,
                                                     label="Axis {0} drift relative".format(x_axis.label)))
        das.append(snomtools.data.datasets.DataArray([x_axis.value_floatindex(x).magnitude for x in self.xdrift],
                                                     unit=x_axis.units,
                                                     label="Axis {0} drift position".format(x_axis.label)))
        das.append(snomtools.data.datasets.DataArray(
            [(x_axis.value_floatindex(x) - x_axis.value_floatindex(self.template_origin[1])).magnitude
             for x in self.xdrift],
            unit=x_axis.units,
            label="Axis {0} drift relative position".format(x_axis.label)))

        return snomtools.data.datasets.DataSet("drift from " + self.data.label, das, [d_axis])

    def as_relative_vectors(self, vectors):
        """
        This is the iterator form of the relative_vector() below.
        """
        for vec in vectors:
            yield self.relative_vector(vec)

    def relative_vector(self, vector):
        """
        Calculate a relative driftvector by subtracting the template_origin vector.

        :param vector: 2-tuple of (y-pixel, x-pixel), the absolute driftvector.

        :return: 2-tuple of (y-pixel-difference, x-pixel-difference), the relative driftvector.
        """
        o_y, o_x = self.template_origin
        d_y, d_x = vector
        return (d_y - o_y, d_x - o_x)

    def generate_shiftvector(self, stack_index):
        """
        Generates the full shift vector according to the shape of self.data (minus the stackAxis) out of the 2d drift
        vectors at a given index along the stackAxis.
        # TESTME: I think the docs here might be wrong and actually the full vector INCLUDING the stackaxis is returned.

        :param int stack_index: An index along the stackAxis

        :return: The 1D array of shift values of length len(self.data)-1.
        :rtype: np.ndarray
        """
        # Initialize empty shiftvector according to the number of dimensions of the data:
        arr = np.zeros(len(self.data.shape))
        # Get the drift at the index position as a numpy array:
        # drift = np.array(list(self.drift_relative)[stack_index])
        drift = np.array(self.relative_vector(self.drift[stack_index]))
        # Put the negated drift in the corresponding shiftvector places:
        np.put(arr, [self.dyAxisID, self.dxAxisID], -drift)
        return arr

    def __getitem__(self, sel):
        """
        Return the shifted data for a selection (slice) of the data.

        :param sel:  A selection (slice) addressing a range of the data.

        :return: DataArray containing the shifted data
        """
        return self.corrected_slice(sel)

    def corrected_slice(self, sel, dtype=None):
        """
        Return the shifted data for a selection (slice) of the data.

        :param sel:  A selection (slice) addressing a range of the data.

        :param dtype: The numeric data type of the corrected data. By default, the type of the original data is kept.
            This means that for an approximation to non-integer values, a float type must be given here.
        :type dtype: numpy.dtype or castable

        :return: DataArray containing the shifted data
        """
        # Get full addressed slice from selection.
        full_selection = full_slice(sel, len(self.data.shape))
        slicebase_wo_stackaxis = np.delete(full_selection, self.dstackAxisID)

        shifted_slice_list = []
        # Iterate over all selected elements along dstackAxis:
        for i in iterfy(np.arange(self.data.shape[self.dstackAxisID])[full_selection[self.dstackAxisID]]):
            # Generate full slice of data to shift, by inserting i into slicebase:
            subset_slice = tuple(np.insert(slicebase_wo_stackaxis, self.dstackAxisID, i))
            # Get shiftvector for the stack element i:
            shift = self.generate_shiftvector(i)
            # Get the shifted data from the Data_Handler method:
            shifted_data = self.data.get_datafield(0).data.shift_slice(subset_slice, shift,
                                                                       order=self.interpolation_order,
                                                                       output=dtype)
            # Attach data to list:
            shifted_slice_list.append(shifted_data)
        if len(shifted_slice_list) < 2:  # We shifted only a single slice along the stackAxis:
            return shifted_slice_list[0]
        else:  # We shifted several slices, so we have to stack them together again.
            return shifted_slice_list[0].__class__.stack(shifted_slice_list)

    def corrected_data(self, h5target=None, dtype=None):
        """
        Return the full driftcorrected dataset.
        2D data is shifted for each position along the stack to negate the drift.
        Shifting is done with DataHandler_H5/_np.shift_slice methods, which have scipy.ndimage.interpolation.shift
        under the hood.
        If h5target is given, the calculations are done chunk-wise for optimal performance.

        :param h5target: A hdf5 target (path for hdf5-File or h5py Group) to write to.

        :param dtype: The numeric data type of the corrected data. By default, the type of the original data is kept.
            This means that for an approximation to non-integer values, a float type must be given here.
        :type dtype: numpy.dtype or castable

        :return: The driftcorrected DataSet.
        """

        oldda = self.data.get_datafield(0)
        if dtype is None:
            dtype = oldda.dtype
        else:
            dtype = np.dtype(dtype)

        if h5target:
            # ToDO:implement chunkwise iteration. e.g. t,E,y,x resolved has chunks (12,6,41,41) with dim (383,81,650,650) = 1.6 GB
            # Probe HDF5 initialization to optimize buffer size:
            chunk_size = snomtools.data.h5tools.probe_chunksize(shape=self.data.shape, dtype=dtype)
            min_cache_size = np.prod(self.data.shape, dtype=np.int64) // self.data.shape[self.dstackAxisID] * \
                             chunk_size[self.dstackAxisID] * dtype.itemsize  # byte size of numeric type.
            use_cache_size = min_cache_size + 256 * 1024 ** 2  # Add 256 MB just to be sure.
            # Initialize data handler to write to:
            dh = snomtools.data.datasets.Data_Handler_H5(unit=str(self.data.datafields[0].units), shape=self.data.shape,
                                                         chunk_cache_mem_size=use_cache_size,
                                                         dtype=dtype)

            # Calculate driftcorrected data and write it to dh:
            if verbose:
                import time
                start_time = time.time()
                print(time.ctime())
                print("Calculating {0} driftcorrected slices...".format(self.data.shape[self.dstackAxisID]))
            # Get full slice for all the data:
            full_selection = full_slice(np.s_[:], len(self.data.shape))
            slicebase_wo_stackaxis = np.delete(full_selection, self.dstackAxisID)
            # Iterate over all elements along dstackAxis:
            for i in range(self.data.shape[self.dstackAxisID]):
                # Generate full slice of data to shift, by inserting i into slicebase:
                subset_slice = tuple(np.insert(slicebase_wo_stackaxis, self.dstackAxisID, i))
                # Get shiftvector for the stack element i:
                shift = self.generate_shiftvector(i)
                if verbose:
                    step_starttime = time.time()
                # Get the shifted data from the Data_Handler method:
                shifted_data = self.data.get_datafield(0).data.shift_slice(subset_slice, shift,
                                                                           order=self.interpolation_order,
                                                                           output=dtype)
                if verbose:
                    print('interpolation done in {0:.2f} s'.format(time.time() - step_starttime))
                    step_starttime = time.time()
                # Write shifted data to corresponding place in dh:
                dh[subset_slice] = shifted_data
                if verbose:
                    print('data written in {0:.2f} s'.format(time.time() - step_starttime))
                    tpf = ((time.time() - start_time) / float(i + 1))
                    etr = tpf * (self.data.shape[self.dstackAxisID] - i + 1)
                    print("Slice {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(i, self.data.shape[
                        self.dstackAxisID], etr, tpf))

            # Initialize DataArray with data from dh:
            newda = snomtools.data.datasets.DataArray(dh, label=oldda.label, plotlabel=oldda.plotlabel,
                                                      h5target=dh.h5target)
        else:
            newda = snomtools.data.datasets.DataArray(self.corrected_slice(np.s_[:], dtype=dtype), label=oldda.label,
                                                      plotlabel=oldda.plotlabel)
        # Put all the shifted data and old axes together to new DataSet:
        newds = snomtools.data.datasets.DataSet(self.data.label + " driftcorrected", (newda,), self.data.axes,
                                                self.data.plotconf, h5target=h5target)
        return newds

    def match_rotation_scale(self, reference_data, tomatch_data, angle_settings=(0, 0, 0), scale_settings=(1, 0, 0),
                             digits=5,
                             output_dir=None, saveImg=False,
                             saveRes=False):
        '''
        Calculates the correlation of different scales and rotations of tomatch_data against some reference_data
        :param reference_data: 2D dataset
        :param tomatch_data: 2D dataset
        :param angle_settings:	tuple (center, resolution, number of variations)
        :param scale_settings:	tuple (center, resolution, number of variations)
        :param digits: number of digits to round the variations to
        :param output_dir: save target
        :param saveImg:	save all rotated and scaled images
        :param saveRes:	save results as TXT with titles
        :return: 	Results array with (angle, scale, ypos, xpos, correlation).

        Example:
            reference_data = ds.DataArray(np.float32(reference_data))
            tomatch_data_cropped = ds.DataArray(crop_around_center(np.float32(tomatch_data), 100, 100))

            match_rotation_scale(reference_data, tomatch_data_cropped, angle_settings=(3, 0.1, 300), output_dir=output_dir, saveRes=True)
        '''
        rot_crop_data = rot.rot_scale_data(data=tomatch_data, angle_settings=angle_settings,
                                           scale_settings=scale_settings, digits=digits, output_dir=output_dir,
                                           saveImg=saveImg)
        results = []
        for variations in rot_crop_data:
            print('Matching: angle ' + str(variations[0]) + ' scale ' + str(variations[1]))
            template = np.float32(rot_crop_data[variations])
            drift_res = Drift.template_matching(reference_data, template, method='cv.TM_CCOEFF_NORMED', subpixel=False)
            results.append((variations[0], variations[1], drift_res[0][0], drift_res[0][1], drift_res[1]))

        results = np.asarray(results)

        if saveRes == True:
            np.savetxt(output_dir + 'rot_scale_max' + str(results[:, 4].max()) + '.txt', np.asarray(results))

        return results

    @classmethod
    def template_matching_stack(cls, data, template, stackAxisID, method='cv.TM_CCOEFF_NORMED', subpixel=True,
                                threshold=(0, 0.2)):
        """
        Passes the data of a 3D array along the stackAxis in form of 2D data to the template_matching function

        :param data: 3D dataset

        :param template: 2D template which is used to calculate the correlation

        :param stackAxisID: Axis along which the 2D data is extracted and passed for template_matching

        :param method: Method to calculate the Correlation between template and data. Possible methods: 'cv.TM_CCOEFF',
            'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

        :param subpixel: Generate subpixel accurate drift vectors

        :param threshold: Threshold of calculated xCorr-values at detected positions that are deemed trustworthy

        :return: List of tuples containing the coordinates of best correlation corrected for values below threshold
        """
        driftlist = []

        if verbose:
            import time
            start_time = time.time()
            print("Calculating {0} driftvectors...".format(data.shape[stackAxisID]))

        for i in range(data.shape[stackAxisID]):
            slicebase = [np.s_[:], np.s_[:]]
            slicebase.insert(stackAxisID, i)
            slice_ = tuple(slicebase)
            driftlist.append(cls.template_matching((data.data[slice_]), template, method, subpixel))

            if verbose:
                tpf = ((time.time() - start_time) / float(i + 1))
                etr = tpf * (data.shape[stackAxisID] - i + 1)
                print("vector {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(i, data.shape[stackAxisID], etr,
                                                                                       tpf))
                print("position ({0:.1f},{1:.1f}) correlation {2:.3f}".format(driftlist[-1][0][0],
                                                                              driftlist[-1][0][1],
                                                                              driftlist[-1][1]))

        indexList = cls.findindex(threshold[0], threshold[1], [result[1] for result in driftlist])
        driftlist_corrected = cls.cleanList(indexList, [xydata[0] for xydata in driftlist])
        return driftlist_corrected

    @staticmethod
    def template_matching(data_to_match, template, method='cv.TM_CCOEFF_NORMED', subpixel=True):
        """
        Uses the openCV matchTemplate function to calculate a correlation between a 2D template array and the 2D data
        array.
        The coordinates of the maximum (or minimum for SQDIFF methods) of the calculated correlation are returned.

        :param data_to_match: 2D dataset

        :param template: 2D template which is used to calculate the correlation

        :param method: Method to calculate the Correlation between template and data. Possible methods: 'cv.TM_CCOEFF',
            'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

        :param subpixel: Generate subpixel accurate drift vectors

        :return: Returns the coordinate (y,x) of the correlation maximum referenced to the top left corner of the
            template and the value of the correlation at that point.
        """
        method = eval(method)

        data_to_match = np.float32(np.array(data_to_match.data))
        template = np.float32(np.array(template))

        res = cv.matchTemplate(data_to_match, template, method)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            xCorrValue = res.min()
            min_loc = np.unravel_index(res.argmin(), res.shape)
            if subpixel:
                top_left = Drift.subpixel_peak(min_loc, res)
            else:
                top_left = min_loc
        else:
            xCorrValue = res.max()
            max_loc = np.unravel_index(res.argmax(), res.shape)

            if subpixel:
                top_left = Drift.subpixel_peak(max_loc, res)
            else:
                top_left = max_loc

        return top_left, xCorrValue

    @staticmethod
    def subpixel_peak(max_var, results):
        """
        Extrapolates the position of a maximum in a 2D array by logarithmic evaluation of it's surrounding values.
        I.e. for x: max_new=max + ( log(f(x-1)) - log(f(x+1)) ) / ( 2*log(f(x-1)) + 2*log(f(x+1)) - 4*log(f(x)) )

        :param max_var: tuple with position of the maximum

        :param results: array with the values in which the subpixel maximum is to be found. Has to contain at least the
            neighbouring elements of max_var.

        :return: tuple of subpixel accurate maximum position
        """
        y = max_var[0]
        x = max_var[1]
        try:
            y_sub = y \
                    + (np.log(results[y - 1, x]) - np.log(results[y + 1, x])) \
                    / \
                    (2 * np.log(results[y - 1, x]) + 2 * np.log(results[y + 1, x]) - 4 * np.log(results[y, x]))
            x_sub = x + \
                    (np.log(results[y, x - 1]) - np.log(results[y, x + 1])) \
                    / \
                    (2 * np.log(results[y, x - 1]) - 4 * np.log(results[y, x]) + 2 * np.log(results[y, x + 1]))
        except (IndexError):
            y_sub = y
            x_sub = x
            print('Warning: Subpixel ignored once. Index out of bounds')
        return (y_sub, x_sub)

    @staticmethod
    def extract_3Ddata(data, stackAxisID, yAxisID, xAxisID):
        """
        Projects the data of a n-D dataset onto the 3D system of stackAxis, yAxis, xAxis.

        :param data:

        :param stackAxisID: Defines the stack axis that is not projected and can be use for template_matching

        :param yAxisID: ID of the first axis of the image, i.e. y

        :param xAxisID: ID of the second axis of the image, i.e. x

        :return: Dataset projected on the defined axes while keeping the stackAxis
        """
        assert isinstance(data, snomtools.data.datasets.DataSet), \
            "ERROR: No dataset or ROI instance given to extract_3Ddata."
        if [data.get_axis_index(n) for n in [stackAxisID, yAxisID, xAxisID]] == [0, 1, 2]:
            return data
        else:
            return data.project_nd(stackAxisID, yAxisID, xAxisID)

    @staticmethod
    def extract_templatedata(data, yAxisID, xAxisID):
        """
        Transforms the data of a template for further processing in the template_matching method.

        :param data: Raw dataset as quantity with axisID's

        :param yAxisID: ID of the first axis, i.e. y

        :param xAxisID: ID of the second axis, i.e. x

        :return: Data of the whole template projected onto the two defined axes.
        """
        assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
            "ERROR: No dataset or ROI instance given to extract_templatedata."

        yAxisID = data.get_axis_index(yAxisID)
        xAxisID = data.get_axis_index(xAxisID)

        return data.project_nd(yAxisID, xAxisID).get_datafield(0)

    @staticmethod
    def guess_templatedata(data, yAxisID, xAxisID):
        """
        Generates a default template for the template_matching method. The template is generated in the center between
            2/5 and 3/5 of the image.

        :param data: Any dataset as quantity with axisID's

        :param yAxisID: ID of the first axis, i.e. y

        :param xAxisID: ID of the second axis, i.e. x

        :return: Projected ROI of the 2/5 to 3/5 field.
        """
        assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
            "ERROR: No dataset or ROI instance given to guess_template."

        yAxisID = data.get_axis_index(yAxisID)
        xAxisID = data.get_axis_index(xAxisID)
        fieldshape = (data.shape[yAxisID], data.shape[xAxisID])
        yl, yr, xl, xr = fieldshape[0] * 2 // 5, fieldshape[0] * 3 // 5, fieldshape[1] * 2 // 5, fieldshape[1] * 3 // 5
        limitlist = {yAxisID: (yl, yr), xAxisID: (xl, xr)}
        roi = snomtools.data.datasets.ROI(data, limitlist, by_index=True)
        return roi.project_nd(yAxisID, xAxisID).get_datafield(0)

    @staticmethod
    def findindex(lowerlim, upperlim, inputlist):
        """Returns List indices for all elements between lower and upper limit"""
        return [n for n, item in enumerate(inputlist) if lowerlim < item and item < upperlim]

    @staticmethod
    def cleanList(indexes, inputlist):
        """Substitutes list[i] with [i-1] for all i in indexes"""
        for i in indexes:
            try:
                inputlist[i] = inputlist[i - 1]
            except (IndexError):
                j = i + 1
                while j in indexes:
                    j = j + 1
                inputlist[i] = inputlist[j]
            except:
                print('Warning: cleanlist failed for Object ' + str(i))
                pass

        return inputlist


class Terra_maxmap(object):
    def __init__(self, data=None, precalculated_map=None, energyAxisID=None, yAxisID=None, xAxisID=None,
                 subpixel=True, method=None, interpolation_order=None, use_meandrift=True, binning=None):

        if energyAxisID is None:
            self.deAxisID = data.get_axis_index('energy')
        else:
            self.deAxisID = data.get_axis_index(energyAxisID)
        if yAxisID is None:
            self.dyAxisID = data.get_axis_index('y')
        else:
            self.dyAxisID = data.get_axis_index(yAxisID)
        if xAxisID is None:
            self.dxAxisID = data.get_axis_index('x')
        else:
            self.dxAxisID = data.get_axis_index(xAxisID)
        if binning is None:
            self.binning = 1
        else:
            self.binning = binning

        self.subpixel = subpixel
        self.data = data

        # check for external drift vectors
        if precalculated_map is not None:
            # ToDO: Insert testing for: "Number of energy shiftvectors unequal to xy dimension of data"
            self.drift = precalculated_map
        else:
            self.method = method
            self.drift = None
            print('No internal maxima map calculation implemented. Please load precalculated map')

        if use_meandrift:
            # calculating the mean Drift for the center half of the image to ignore wrong values at edges
            lower0 = np.int_(np.shape(self.drift[:][0])[0] / 4)
            upper0 = np.int_(np.shape(self.drift[:][0])[0] * 3 / 4)
            lower1 = np.int_(np.shape(self.drift[0][:])[0] / 4)
            upper1 = np.int_(np.shape(self.drift[0][:])[0] * 3 / 4)
            self.meanDrift = np.rint(np.mean(self.drift[lower0:upper0, lower1:upper1])).astype(int)

        if interpolation_order is None:
            self.interpolation_order = 0
        else:
            self.interpolation_order = interpolation_order

    @property
    def drift_relative(self):
        return self.as_relative_vectors(self.drift)

    def as_relative_vectors(self, vectors):
        for vec in vectors:
            yield self.relative_vector(vec)

    def relative_vector(self, vector):
        # Subtract mean value from current drift value
        o_E = self.meanDrift
        d_E = vector
        return (d_E - o_E)

    def generate_shiftvector(self, stack_index):
        """
        Generates the full shift vector according to the shape of self.data (minus the stackAxis) out of the drift
            vectors at a given index along the stackAxis.

        :param int stack_index: An index along the stackAxis

        :return: The 1D array of shift values of length len(self.data)-1.
        :rtype: np.ndarray
        """
        # Initialize empty shiftvector according to the number of dimensions of the data:
        arr = np.zeros(len(self.data.shape))
        # Get the drift at the index position as a numpy array:
        drift = np.array(self.relative_vector(self.drift[stack_index[0], stack_index[1]]))
        # Put the negated drift in the corresponding shiftvector place for the energy axis:
        np.put(arr, [self.deAxisID], -drift / self.binning)
        return arr

    def corrected_data(self, h5target=None):
        """Return the full dataset with maxima-map corrected data. Therefore in each xy pixel the data gets shifted along the energy axis"""

        # Address the DataArray with all the data
        fulldata = self.data.get_datafield(0)
        assert isinstance(fulldata, snomtools.data.datasets.DataArray)

        if h5target:
            # --- Prepare data to iterable slices in chunks, calculate driftcorrected data and write it to dh ---:

            # Probe HDF5 initialization to optimize buffer size for xy chunk along full energy and time axis:
            chunk_size = snomtools.data.h5tools.probe_chunksize(shape=self.data.shape)
            min_cache_size = np.prod(self.data.shape, dtype=np.int64) // (self.data.shape[self.dxAxisID]) // \
                             (self.data.shape[self.dyAxisID]) * chunk_size[self.dxAxisID] * \
                             chunk_size[self.dyAxisID] * 4  # 32bit floats require 4 bytes.
            use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 128 MB just to be sure.

            # Initialize data handler to write to:
            dh = snomtools.data.datasets.Data_Handler_H5(unit=str(self.data.datafields[0].units), shape=self.data.shape,
                                                         chunk_cache_mem_size=use_cache_size)

            if verbose:
                import time
                start_time = time.time()
                print(time.ctime())
                xychunks = self.data.shape[self.dxAxisID] * self.data.shape[self.dyAxisID] // fulldata.data.chunks[
                    self.dyAxisID] // fulldata.data.chunks[self.dxAxisID]
                chunks_done = 0
                print("Calculating {0} driftcorrected slices...".format(xychunks))
            # Get full slice for all the data in the xy chunk:
            full_selection = full_slice(np.s_[:], len(self.data.shape))
            # Delete y Axis to prepare insertion of iteration variable for y
            slicebase_wo_yaxis = np.delete(full_selection, self.dyAxisID)

            # Create a cache array with the full size in energy and time axis, therefore remove xy from fulldata.shape
            datasize = list(fulldata.shape)
            xy_indexes = [self.dyAxisID, self.dxAxisID]
            xy_indexes.sort()
            xy_indexes.reverse()
            for dimension in xy_indexes:
                datasize.pop(dimension)
            # Cache array is later used for every xy to cache the shifted data of each xy pixel's stack
            cache_array = np.empty(shape=tuple(datasize), dtype=np.float32)

            # Work on the slices that are contained in the same chunk for xy -> fast
            for chunkslice in fulldata.data.iterchunkslices(dims=(self.dyAxisID, self.dxAxisID)):
                if verbose:
                    step_starttime = time.time()

                # Create big cache array in which the calculated cache arrays will be buffered so only one write process per chunk occurs ->fast
                bigger_cache_array = np.empty(shape=sliced_shape(chunkslice, fulldata.shape), dtype=np.float32)
                # Address the full data of the chunkslice as numpy array
                fulldata_chunk = snomtools.data.datasets.Data_Handler_np(fulldata.data.ds_data[chunkslice],
                                                                         fulldata.get_unit())
                # define yslice as y axis in chunkslice
                yslice = chunkslice[self.dyAxisID]
                assert isinstance(yslice, slice)
                # find end of y-data: either end of slice or end of y-axis, if yslice.stop is not defined
                if yslice.stop is None:
                    upper_lim = fulldata.shape[self.dyAxisID]
                else:
                    upper_lim = yslice.stop

                # Iterate over all elements along dyAxis in the chunkslice
                for i in range(yslice.start, upper_lim):
                    # Inserting i as iterator to slicebase without yaxis:
                    intermediate_slice = np.insert(slicebase_wo_yaxis, self.dyAxisID, i)
                    # Create a slice with relative coordinates. "yslice.start" is the absolute position of the data and "i - yslice.start" the relative position in the slice
                    intermediate_slice_relative = np.insert(slicebase_wo_yaxis, self.dyAxisID, i - yslice.start)

                    # Delete x Axis analogous to y axis earlier
                    slicebase_wo_xyaxis = np.delete(intermediate_slice, self.dxAxisID)
                    slicebase_wo_xyaxis_relative = np.delete(intermediate_slice_relative, self.dxAxisID)

                    # define xslice as x axis in chunkslice
                    xslice = chunkslice[self.dxAxisID]
                    assert isinstance(xslice, slice)
                    # find end of x-data: either end of slice or end of x-axis, if xslice.stop is not defined
                    if xslice.stop is None:
                        upper_lim = fulldata.shape[self.dxAxisID]
                    else:
                        upper_lim = xslice.stop

                    # Iterate over all elements along dxAxisin the chunkslice:
                    for j in range(xslice.start, upper_lim):
                        # subset_slice = tuple(np.insert(slicebase_wo_xyaxis, self.dxAxisID, j))

                        # Insert "j-xslice.start" as relative iteration variable at the x-Axis position in the slice
                        subset_slice_relative = tuple(
                            np.insert(slicebase_wo_xyaxis_relative, self.dxAxisID, j - xslice.start))

                        # Get shiftvector for the stack element at y,x coordinates i,j:
                        shift = self.generate_shiftvector((i, j))

                        if self.subpixel:
                            # -- calculate shifted data via .shift_slice --
                            # Get the shifted data from the Data_Handler method and put it to cache array:
                            fulldata_chunk.shift_slice(subset_slice_relative, shift, output=cache_array,
                                                       order=self.interpolation_order)

                            # Write shifted data to corresponding place in the bigger cache array:
                            bigger_cache_array[subset_slice_relative] = cache_array
                        else:
                            # -- calculate shifted data via shifted numpy arrays. Only int shift --

                            # cast shift in the coordinate of the energy axis to int
                            shift = np.rint(shift[self.deAxisID]).astype(int)

                            if shift == 0:
                                # if shift=0 write data in the subset_slice_relative to bigger cache array
                                bigger_cache_array[subset_slice_relative] = fulldata_chunk.magnitude[
                                    subset_slice_relative]
                            else:
                                # create slices to cut out the kept data, address it's target position and fill the rest with Nan
                                sourceslice = list(subset_slice_relative)
                                targetslice = list(subset_slice_relative)
                                restslice = list(subset_slice_relative)

                                # Since energy axis is shifted, the slices are changed in the deAxisID axis

                                if shift < 0:
                                    # shift <0 -> data has to be shifted down
                                    s = abs(shift)
                                    sourceslice[self.deAxisID] = np.s_[s:]  # data starting from shift to end is kept
                                    targetslice[self.deAxisID] = np.s_[
                                                                 :-s]  # data should be in the slice starting at 0 ending at end-s
                                    restslice[self.deAxisID] = np.s_[-s:]  # positions end-s until end should be Nan
                                else:
                                    s = abs(shift)
                                    # shift >0 -> data has to be shifted up
                                    sourceslice[self.deAxisID] = np.s_[:-s]  # data starting from 0 to end-s is kept
                                    targetslice[self.deAxisID] = np.s_[s:]  # data should start at s
                                    restslice[self.deAxisID] = np.s_[:s]  # empty space from 0 to s should be Nan

                                # Remove x and y dimension so the size fits
                                for dimension in xy_indexes:
                                    targetslice.pop(dimension)
                                    restslice.pop(dimension)
                                # Write the data using the generated slices for addressing the source in fulldata and the target in cache_array
                                cache_array[tuple(restslice)] = np.nan  # write Nan to restslice positions
                                cache_array[tuple(targetslice)] = fulldata_chunk.magnitude[
                                    tuple(sourceslice)]  # write data from sourceslice to positions of targetslice

                                # Write cache_array to it's subset_slice_relative position in the bigger_cache_array
                                bigger_cache_array[subset_slice_relative] = cache_array

                # After the whole chunkslice is shifted, pass it to the h5 data handler
                dh[chunkslice] = bigger_cache_array
                if verbose:
                    chunks_done += 1
                    print('data interpolated and written in {0:.2f} s'.format(time.time() - step_starttime))
                    tpf = ((time.time() - start_time) / float(chunks_done))
                    etr = tpf * (xychunks - chunks_done)
                    print("Slice {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(chunks_done, xychunks, etr,
                                                                                          tpf))

            # Initialize DataArray with data from dh:
            newda = snomtools.data.datasets.DataArray(dh, label=fulldata.label, plotlabel=fulldata.plotlabel,
                                                      h5target=dh.h5target)

        # if no h5target is given:
        else:
            newda = snomtools.data.datasets.DataArray(self[:], label=fulldata.label, plotlabel=fulldata.plotlabel)

        # Put all the shifted data and old axes together to new DataSet:
        newds = snomtools.data.datasets.DataSet(self.data.label + " maximacorrected", (newda,), self.data.axes,
                                                self.data.plotconf, h5target=h5target)
        return newds


if __name__ == '__main__':  # Testing...
    # testfolder = "test/Drifttest/new"

    import snomtools.data.datasets
    import os

    driftfile = ('fit_maximum2.maxima.min1040max1184.matrix')
    precal_map = np.loadtxt(driftfile)

    objects = os.listdir('rawdata/')
    rawdatalist = []
    for i in objects:
        if i.endswith("Durchlauf.hdf5"):
            rawdatalist.append(i)

    for run in rawdatalist:
        data = snomtools.data.datasets.DataSet.from_h5file('rawdata/' + run, h5target=run + '_testdata.hdf5',
                                                           chunk_cache_mem_size=2048 * 1024 ** 2)

        # data = snomtools.data.datasets.stack_DataSets(data, snomtools.data.datasets.Axis([1, 2, 3], 's', 'faketime'))

        data.saveh5()

        drift = Terra_maxmap(data, precal_map, subpixel=True, binning=16)
        # Calculate corrected data:
        correcteddata = drift.corrected_data(h5target='Maximamap/' + run)
        correcteddata.saveh5()
        print("done.")

    # data = snomtools.data.datasets.DataSet.from_h5file('Maximamap/' + run, h5target=run + '_testdata.hdf5',
    # 												   chunk_cache_mem_size=2048 * 1024 ** 2)
    #
    # # data = snomtools.data.datasets.stack_DataSets(data, snomtools.data.datasets.Axis([1, 2, 3], 's', 'faketime'))
    #
    # data.saveh5()
    #
    # driftfile = ('Summenbilder/' + run.replace('.hdf5', '.txt'))
    #
    # precal_drift = np.loadtxt(driftfile)
    # precal_drift = [tuple(row) for row in precal_drift]
    #
    # drift = Drift(data, precalculated_drift=precal_drift, stackAxisID="delay", template=None, subpixel=True,
    # 			  template_origin=(123, 347))
    #
    # # Calculate corrected data:
    # correcteddata = drift.corrected_data(h5target='Maximamap/Driftcorrected/' + run)
    #
    # correcteddata.saveh5()

    print("done.")
    print("the end.")
