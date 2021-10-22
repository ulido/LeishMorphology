# Leishmania Morphology Image Analysis Pipeline

This pipeline works on three-channel microscopy images of fixed *Leishmania mexicana* cells - a phase contrast image, a DAPI channel and an mNeonGreen gene channel. The segmentation is performed on the phase image, with some input from the DAPI channel (the location of the kinetoplast).

## Quickstart

The component doing the image segmentation and analysis is the `ImageSet` class, which takes a path to a directory containing the three channel tif files. It's `identified_cells` property then contains a `SegmentedCellCollection` with a list of `SegmentedCell`s. A `SegmentedCell` object holds information on a single observed cell, such as length, width, flagellum length and a thumbnail image.

```python
import leishmorphology

im = leishmorphology.ImageSet(path_to_imageset)
cells = im.identified_cells
```

To analyse a collection of image sets, one can use the `SegmentedCellCollectionSet` class, which also allows saving of the produced data in a HDF5 file (via `pandas` and `pytables`):

```python
import leishmorphology

cellset = leishmorphology.SegmentedCellCollectionSet([
    leishmorphology.segmentation.ImageSet(path).identified_cells for path in [path1, path2, path3])
cellset.to_hdf("cells.h5")
```

Loading of a collection works via the static method `SegmentedCellCollectionSet.load_hdf`:
```python
import leishmorphology

cellset = leishmorphology.SegmentedCellCollectionSet.load_hdf(path_to_hdf5_file)
```

## Displaying data

The `SegmentedCell`, `SegmentedCellCollection` and `SegmentedCellCollectionSet` classes have rich display methods which integrate into IPython's/Jupyter's rich display infrastructure. Therefore the following happens:
 * A `SegmentedCell` instance displays its thumbnail with the segmented cell and (if found) its flagellum highlighted. It also displays a box with cell measurements.
 * A `SegmentedCellCollection` instance displays a floating grid of the `SegmentedCell`s it contains.
 * A `SegmentedCellCollectionSet` instance displays a scatter plot of cell width vs length (with markers distinguishing cells from different `SegmentedCellCollection`s) and colour showing the flagellum length (if present).

`SegmentedCell` is a python `dataclass` with each attribute describing a property of the segmented cell. `SegmentedCellCollection` has a `name` attribute which, if not given, defaults to the image set path. It contains a `cell_list` attribute which simply holds a list of `SegmentedCells`. Finally, `SegmentedCellCollectionSet` has a single attribute `sets` which holds a list of `SegmentedCellCollection`s. The latter also has a method `to_dataframe` which returns a `pandas.DataFrame` table with one cell per row.
