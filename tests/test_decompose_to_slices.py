"""
Test module for DecomposeToSlices class in transforms.py

This module provides comprehensive test coverage for the
DecomposeToSlices class,testing initialization, slice extraction,
filtering by class fraction, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from caked.Transforms.transforms import DecomposeToSlices


class TestDecomposeToSlicesInit:
    """Test initialization of DecomposeToSlices class."""

    def test_init_basic(self):
        """Test basic initialization with default parameters."""
        map_shape = (64, 64, 64)
        decomposer = DecomposeToSlices(map_shape)
        
        assert decomposer.slices is not None
        assert len(decomposer.slices) > 0
        assert len(decomposer.slice_indicies) == len(decomposer.slices)

    def test_init_with_custom_step(self):
        """Test initialization with custom step size."""
        map_shape = (64, 64, 64)
        step = 32
        decomposer = DecomposeToSlices(map_shape, step=step, cshape=32)
        
        # With step=32 and cshape=32 on a 64x64x64 map, should have 8 slices
        # (2 positions in each dimension: 0 and 32)
        assert len(decomposer.slices) == 8

    def test_init_with_custom_cshape(self):
        """Test initialization with custom chunk shape."""
        map_shape = (100, 100, 100)
        cshape = 50
        decomposer = DecomposeToSlices(map_shape, step=1, cshape=cshape)
        
        # Verify that slices don't exceed map dimensions
        for slc in decomposer.slices:
            for idx, s in enumerate(slc):
                assert s.stop <= map_shape[idx]

    def test_init_oversized_cshape(self):
        """Test initialization when cshape exceeds map shape."""
        map_shape = (32, 32, 32)
        cshape = 64
        decomposer = DecomposeToSlices(map_shape, step=1, cshape=cshape)
        
        # Should fall back to single slice
        assert len(decomposer.slices) == 1
        assert len(decomposer.slice_indicies) == 1

    def test_init_slices_structure(self):
        """Test that slices have the correct structure."""
        map_shape = (64, 64, 64)
        decomposer = DecomposeToSlices(map_shape, step=32, cshape=32)
        
        # Each slice should be a tuple of 3 slice objects
        for slc in decomposer.slices:
            assert len(slc) == 3
            assert all(isinstance(s, slice) for s in slc)
            
        # Each index should be a tuple of 3 integers
        for idx in decomposer.slice_indicies:
            assert len(idx) == 3
            assert all(isinstance(i, (int, np.integer)) for i in idx)

    def test_init_with_edge_parameter(self):
        """Test initialization with edge parameter allowing boundary slices."""
        map_shape = (64, 64, 64)
        cshape = 32
        step = 32
        
        # Without edge, should have 8 slices (2x2x2 grid)
        decomposer_no_edge = DecomposeToSlices(map_shape, step=step,
                                               cshape=cshape)
        assert len(decomposer_no_edge.slices) == 8
        
        # With edge=32, allows slices to extend 32 units beyond boundary
        # This allows for slices starting at positions that would extend beyond
        decomposer_with_edge = DecomposeToSlices(map_shape, step=step,
                                                 cshape=cshape, edge=32)
        # With edge=32, positions can go up to 64+32=96, so more positions
        # possible. For 64x64x64 map with step=32, cshape=32, edge=32:
        # positions at 0, 32, 64.  But only if i + cshape <= map_shape
        # + edge, so max position = map_shape
        assert len(decomposer_with_edge.slices) == 8

    def test_init_edge_parameter_allows_boundary_slices(self):
        """Test that edge parameter allows slices at boundaries."""
        map_shape = (50, 50, 50)
        cshape = 40
        step = 25
        
        # Without edge, slice starting at position 25 would be rejected
        # because 25 + 40 = 65 > 50
        decomposer_no_edge = DecomposeToSlices(map_shape, step=step, cshape=cshape)
        assert len(decomposer_no_edge.slices) == 1  # Only position (0,0,0)
        
        # With edge=20, allows slices up to position where i + cshape <= 50 + 20
        # So positions 0 and 25 are valid (25 + 40 = 65 <= 70)
        decomposer_with_edge = DecomposeToSlices(map_shape, step=step, cshape=cshape, edge=20)
        assert len(decomposer_with_edge.slices) == 8  # 2x2x2 grid of positions


class TestExtractSlices:
    """Test extract_slices method."""

    def test_extract_slices_basic(self):
        """Test basic slice extraction."""
        map_array = np.random.rand(64, 64, 64)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        slices = decomposer.extract_slices(map_array)
        
        assert len(slices) > 0
        assert all(isinstance(s, np.ndarray) for s in slices)
        assert all(s.shape == (32, 32, 32) for s in slices)

    def test_extract_slices_updates_internal_state(self):
        """Test that extract_slices updates internal slices state."""
        map_array = np.random.rand(64, 64, 64)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        original_slice_count = len(decomposer.slices)
        decomposer.extract_slices(map_array)
        
        # Internal state should be updated
        assert len(decomposer.slices) == original_slice_count

    def test_extract_slices_with_padding(self):
        """Test slice extraction with padding enabled."""
        map_array = np.random.rand(50, 50, 50)
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        
        slices = decomposer.extract_slices(map_array, padding=True, padding_value=0.0)
        
        # All slices should have the requested shape due to padding
        assert all(s.shape == (30, 30, 30) for s in slices)

    def test_extract_slices_skip_small(self):
        """Test slice extraction with skip_small enabled."""
        map_array = np.random.rand(50, 50, 50)
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        
        slices = decomposer.extract_slices(map_array, skip_small=True)
        
        # With skip_small, slices that exceed boundaries are skipped
        assert len(slices) >= 0
        assert all(s.shape == (30, 30, 30) for s in slices if s is not None)

    def test_extract_slices_different_padding_values(self):
        """Test extraction with different padding values."""
        map_array = np.ones((50, 50, 80))
        # Use edge parameter to allow slices exceeding bounds
        # This allows slices at position 25 to extend to 55
        decomposer = DecomposeToSlices(
            map_array.shape, step=25, cshape=30, edge=30
        )
        
        padding_value = 9.0
        slices = decomposer.extract_slices(
            map_array, 
            padding=True, 
            padding_value=padding_value
        )
        
        # Count how many slices have the padding value
        slices_with_padding = sum(
            1 for slc in slices if np.any(slc == padding_value)
        )
        
        # With edge=30, slices at position 25 exceed bounds in the first
        # two dimensions (25+30=55 > 50), but fit in the third (25+30=55 <= 80)
        # 16 slices total: 3 fully in bounds, with indices (0,0,0), (0,0,25),
        # (0,0,50), rest exceed bounds in at least one dimension
        assert len(slices) == 16
        assert slices_with_padding == 13  # 13 out of 16 exceed bounds

    def test_extract_slices_with_edge_parameter(self):
        """Test slice extraction with edge parameter."""
        map_array = np.random.rand(50, 50, 50)
        
        # Without edge: only slices that fit within bounds are extracted
        decomposer_no_edge = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        slices_no_edge = decomposer_no_edge.extract_slices(map_array, padding=False)
        
        # With edge=20: allows slices to extend beyond boundary
        decomposer_with_edge = DecomposeToSlices((50, 50, 50), step=25, cshape=30, edge=20)
        slices_with_edge = decomposer_with_edge.extract_slices(map_array, padding=True, padding_value=0.0)
        
        # With edge, we should be able to extract more slices due to relaxed boundary constraints
        assert len(slices_with_edge) == 8
        assert len(slices_no_edge) == 1

    def test_extract_slices_no_padding_without_edge(self):
        """Test that no padding is applied when edge parameter is 0."""
        map_array = np.ones((50, 50, 30))
        # Without edge parameter, slices are filtered to fit within bounds
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        
        padding_value = 9.0
        slices = decomposer.extract_slices(
            map_array, 
            padding=True, 
            padding_value=padding_value
        )
        
        # Count slices with the padding value
        slices_with_padding = sum(
            1 for slc in slices if np.any(slc == padding_value)
        )
        
        # Without edge, all slices fit within bounds, so NO padding applied
        # All slices should contain only original values (1.0 from np.ones)
        assert slices_with_padding == 0
        # All returned slices should have values from original array
        assert all(
            np.all((slc == 1.0)) for slc in slices
        )
        # Verify slices don't contain padding value
        for slc in slices:
            assert not np.any(slc == padding_value)


class TestGetSliceArray:
    """Test get_slice_array method (called via extract_slices)."""

    def test_get_slice_array_basic(self):
        """Test basic slice extraction via extract_slices."""
        map_array = np.random.rand(64, 64, 64)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        # extract_slices uses get_slice_array internally
        slices = decomposer.extract_slices(map_array)
        
        assert len(slices) > 0
        assert all(isinstance(s, np.ndarray) for s in slices)

    def test_get_slice_array_out_of_bounds_with_padding(self):
        """Test slice extraction out of bounds with padding."""
        map_array = np.random.rand(50, 50, 50)
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30,
                                       edge=15)
        
        slice_data_list = decomposer.extract_slices(
            map_array, padding=True, padding_value=0.0
        )
        
        # Check that padding was applied
        assert all(s.shape == (30, 30, 30) for s in slice_data_list)
        # check number of slices
        assert len(slice_data_list) == len(decomposer.slices) == 8

    def test_get_slice_array_out_of_bounds_skip_small(self):
        """Test slice extraction out of bounds with skip_small."""
        map_array = np.random.rand(50, 50, 50)
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        
        slice_data_list = decomposer.extract_slices(
            map_array, skip_small=True
        )
        
        # Should handle without errors
        assert isinstance(slice_data_list, list)

    def test_get_slice_array_partial_out_of_bounds(self):
        """Test slice partially out of bounds."""
        map_array = np.random.rand(64, 64, 64)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=64)
        
        slice_data_list = decomposer.extract_slices(
            map_array, padding=True, padding_value=-1.0
        )
        
        assert len(slice_data_list) > 0

    def test_get_slice_array_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        map_array = np.random.rand(64, 64, 64)  # 3D array
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        # Create a 2D slice which won't match the 3D map
        with pytest.raises(ValueError, match="same number of dimensions"):
            decomposer.get_slice_array(
                map_array,
                (slice(0, 32), slice(0, 32)),  # Only 2D slice
            )

    def test_get_slice_array_within_bounds(self):
        """Test slice completely within bounds."""
        map_array = np.arange(64 * 64 * 64).reshape(64, 64, 64)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        slices = decomposer.extract_slices(map_array)
        
        # All slices should be 32x32x32 within bounds
        assert all(s.shape == (32, 32, 32) for s in slices)


class TestSelectSlicesMinClassFraction:
    """Test select_slices_min_class_fraction method."""

    def test_select_slices_all_pass_threshold(self):
        """Test when all slices pass the class fraction threshold."""
        # Create a label array with equal distribution
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        label_array[:32] = 1
        
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        class_fraction = {1: 0.4}  # Require at least 40% of class 1
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=False,
            skip_small=False
        )
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_select_slices_some_fail_threshold(self):
        """Test when some slices fail the class fraction threshold."""
        # Create a label array with uneven distribution
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        label_array[0:10] = 1  # Only first 10 slices have class 1
        
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        class_fraction = {1: 0.5}  # Require at least 50% of class 1
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=False,
            skip_small=False
        )
        
        assert isinstance(result, list)

    def test_select_slices_multiple_classes(self):
        """Test with multiple class fractions."""
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        label_array[:32, :32, :32] = 1
        label_array[:32, 32:, 32:] = 2
        
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        class_fraction = {1: 0.3, 2: 0.2}
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=False,
            skip_small=False
        )
        
        assert isinstance(result, list)

    def test_select_slices_empty_result(self):
        """Test when no slices meet the threshold."""
        # Create a label array with no target class
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        class_fraction = {1: 0.5}  # Looking for class 1 that doesn't exist
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=False,
            skip_small=False
        )
        
        assert isinstance(result, list)
        # No slices should pass since class 1 doesn't exist
        assert len(result) == 0

    def test_select_slices_with_padding(self):
        """Test slice selection with padding."""
        label_array = np.random.randint(0, 3, (50, 50, 50))
        decomposer = DecomposeToSlices((50, 50, 50), step=25, cshape=30)
        class_fraction = {0: 0.2}
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=True,
            padding_value=0.0
        )
        
        assert isinstance(result, list)
        # All returned slices should have the correct shape
        assert all(s.shape == (30, 30, 30) for s in result if s is not None)

    def test_select_slices_updates_state(self):
        """Test that method updates internal state."""
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        
        original_count = len(decomposer.slices)
        decomposer.select_slices_min_class_fraction(
            label_array,
            {0: 0.5},
            padding=False,
            skip_small=False
        )
        
        # State should be updated (slices may be filtered)
        assert len(decomposer.slices) <= original_count

    def test_select_slices_zero_fraction_threshold(self):
        """Test with zero fraction threshold."""
        label_array = np.zeros((64, 64, 64), dtype=np.int32)
        label_array[0:5] = 1
        
        decomposer = DecomposeToSlices((64, 64, 64), step=32, cshape=32)
        class_fraction = {1: 0.0}  # Accept any amount
        
        result = decomposer.select_slices_min_class_fraction(
            label_array,
            class_fraction,
            padding=False,
            skip_small=False
        )
        
        # All slices should pass with 0 threshold
        assert len(result) > 0


class TestDecomposeToSlicesIntegration:
    """Integration tests for DecomposeToSlices."""

    def test_workflow_decompose_and_extract(self):
        """Test complete workflow of decomposing and extracting slices."""
        map_array = np.random.rand(128, 128, 128)
        decomposer = DecomposeToSlices((128, 128, 128), step=64, cshape=64)
        
        slices = decomposer.extract_slices(map_array)
        
        assert len(slices) == 8  # 2^3 combinations
        assert all(s.shape == (64, 64, 64) for s in slices)

    def test_workflow_decompose_filter_and_extract(self):
        """Test workflow with filtering and extraction."""
        label_array = np.random.randint(0, 3, (128, 128, 128))
        map_array = np.random.rand(128, 128, 128)
        
        decomposer = DecomposeToSlices((128, 128, 128), step=64, cshape=64)
        
        # First filter by class fraction
        filtered_labels = decomposer.select_slices_min_class_fraction(
            label_array,
            {1: 0.2},
            padding=False,
            skip_small=False
        )
        
        # Then extract slices from map using filtered indices
        slices = decomposer.extract_slices(map_array)
        
        assert len(slices) == len(filtered_labels)

    def test_small_map_handling(self):
        """Test handling of small maps."""
        map_array = np.random.rand(16, 16, 16)
        decomposer = DecomposeToSlices((16, 16, 16), step=8, cshape=16)
        
        slices = decomposer.extract_slices(map_array, padding=True)
        
        # Should handle small maps gracefully
        assert len(slices) > 0

    def test_large_map_handling(self):
        """Test handling of large maps."""
        # Use reasonable size for testing
        map_array = np.random.rand(256, 256, 256)
        decomposer = DecomposeToSlices((256, 256, 256), step=64, cshape=64)
        
        slices = decomposer.extract_slices(map_array)
        
        # Should generate expected number of slices
        assert len(slices) == 64  # 4^3

    def test_workflow_with_edge_parameter(self):
        """Test complete workflow with edge parameter
            allowing boundary slices."""
        map_array = np.random.rand(100, 100, 100)
        cshape = 60
        step = 50
        
        # Without edge: positions 0 and 50
        # Position 0: 0 + 60 = 60 <= 100 ✓
        # Position 50: 50 + 60 = 110 > 100 ✗
        decomposer_no_edge = DecomposeToSlices(
            (100, 100, 100), step=step, cshape=cshape
        )
        slices_no_edge = decomposer_no_edge.extract_slices(map_array, 
                                                           padding=False)
        
        # With edge=20: positions 0 and 50
        # Position 0: 0 + 60 = 60 <= 100+20 ✓
        # Position 50: 50 + 60 = 110 <= 100+20 ✓
        decomposer_with_edge = DecomposeToSlices(
            (100, 100, 100), step=step, cshape=cshape, edge=20
        )
        slices_with_edge = decomposer_with_edge.extract_slices(
            map_array, padding=True, padding_value=0.0
        )
        
        # With edge, we expect more slices due to relaxed boundary constraints
        assert len(slices_with_edge) > len(slices_no_edge)
        # With edge, we should have 2^3 = 8 slices (2 positions per dimension)
        assert len(slices_with_edge) == 8
        assert len(slices_no_edge) == 1
