# Leishmania Morphology Image Analysis Pipeline

This pipeline works on three-channel microscopy images of fixed /Leishmania mexicana/ cells - a phase contrast image, a DAPI channel and an mNeonGreen gene channel. The segmentation is performed on the phase image, with some input from the DAPI channel (the location of the kinetoplast).

## Quickstart

The component doing the image segmentation and analysis is the `ImageSet` class, which takes a path to a directory containing the three channel tif files. It's `identified_cells` property then contains a `SegmentedCellCollection` with a list of `SegmentedCell`s. A `SegmentedCell` object holds information on a single observed cell, such as length, width, flagellum length and a thumbnail image.

```python
import leishmorphology

im = leishmorphology.ImageSet(path_to_imageset)
cells = im.identified_cells
```
