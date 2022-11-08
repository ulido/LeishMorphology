import pathlib
import re
from collections.abc import Mapping
import matplotlib.pyplot as plt
import tifffile
import io
from tqdm.auto import tqdm
import xmltodict

from .datastructures import SegmentedCellCollectionSet
from .segmentation import ImageSet
from .parallelprocess import _analyse_image_path_collection

class DirectoryStructureError(Exception):
    def __init__(self, name):
        self.name = name

class PlateDirectoryNameError(DirectoryStructureError):
    def __str__(self):
        return f'Directory name "{self.name}" is not a valid plate directory name (such as <Plate ID>_<PCR Date>).'

class StageDirectoryNameError(DirectoryStructureError):
    def __str__(self):
        return f'Directory name "{self.name}" is not a valid stage name (such as "Axa" or "Pro").'
    
class SubPlateDirectoryNameError(DirectoryStructureError):
    def __str__(self):
        return f'Directory name "{self.name}" is not a valid sub plate name (such as "PlateA").'

class SubPlateWellImagingReplicateDirectoryNameError(DirectoryStructureError):
    def __str__(self):
        return f'Directory name "{self.name}" is not a valid well/imaging replicate name (such as <extra info>_AA1_1).'
    
class NoImagesFoundError(Exception):
    def __init__(self, well):
        self.well = well
    
    def __str__(self):
        return f'No image files found in well {self.well}.'

class Directory(Mapping):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._validate()
        self._recurse()
        
    def _validate(self):
        raise NotImplementedError

    def _recurse(self):
        raise NotImplementedError
        
    def __str__(self):
        s = repr(self)
        for _, child in sorted(self.children.items(), key=lambda x: x[0]):
            s += '\n  ' + str(child).replace('\n', '\n  ')
        return s
        
    def __getitem__(self, key):
        return self.children[key]
    
    def __iter__(self):
        return iter(self.children)
    
    def __len__(self):
        return len(self.children)
    
    def _ipython_key_completions_(self):
        return self.children.keys()
        
class PlateDirectory(Directory):
    plate_regex = re.compile('(?P<name>[A-Z0-9]+)_(?P<pcr_date>[0-9]{8})')
        
    def _validate(self):
        self._validate_name()
        
    def _recurse(self):
        self.children = {}
        for stagedir in [d for d in self.path.iterdir() if d.is_dir()]:
            stage = StageDirectory(stagedir)
            self.children[stage.stage_name] = stage
            
    def _validate_name(self):
        folder_name = self.path.name
        m = self.plate_regex.match(folder_name)
        if m is None:
            raise PlateDirectoryNameError(folder_name)
        
        self.plate_name = m.group('name')
        self.pcr_date = m.group('pcr_date')
        
    def __repr__(self):
        return f'<Plate directory "{self.plate_name}" (Date {self.pcr_date})>'
    
class StageDirectory(Directory):
    valid_stage_names = {'Axa', 'Pro'}
    
    def _validate(self):
        folder_name = self.path.name
        if folder_name not in self.valid_stage_names:
            raise StageDirectoryNameError(folder_name)
        self.stage_name = folder_name

    def _recurse(self):
        self.children = {}
        for subplatedir in [d for d in self.path.iterdir() if d.is_dir()]:
            subplate = SubPlateDirectory(subplatedir)
            self.children[subplate.subplate_name[-1]] = subplate
    
    def analyse_images(self):
        if self.stage_name == 'Axa':
            raise NotImplementedError('Image analysis for amastigote stage is not implemented yet.')
        paths_names = [(well.path, name)
                       for subplate in self.children.values()
                       for name, well in subplate.items()]
        return SegmentedCellCollectionSet(_analyse_image_path_collection(paths_names))
    
    def __repr__(self):
        return f'<{self.stage_name} stage>'

class SubPlateDirectory(Directory):
    valid_subplate_names = {'PlateA', 'PlateB', 'PlateC', 'PlateD'}
    
    def _validate(self):
        folder_name = self.path.name
        if folder_name not in self.valid_subplate_names:
            raise SubPlateDirectoryNameError(folder_name)
        self.subplate_name = folder_name

    def _recurse(self):
        self.children = {}
        for welldir in [d for d in self.path.iterdir() if d.is_dir()]:
            well = SubPlateWellImagingReplicateDirectory(welldir)
            if well.subplate != self.subplate_name[-1]:
                raise SubPlateNameMismatch(self.subplate_name, self.well)
            try:
                well._tif_filename
            except NoImagesFoundError:
                continue
            self.children[well.row + well.column + '_' + well.replicate] = well
    
    def __repr__(self):
        return f'<Subplate {self.subplate_name[-1]}>'

class SubPlateWellImagingReplicateDirectory(Directory):
    name_regex = re.compile('(?P<extra_info>.*)_(?P<subplate>[A-D])(?P<row>[A-D])(?P<column>[0-6])_(?P<replicate>[0-9]+)')
    valid_subplate_names = {'PlateA', 'PlateB', 'PlateC', 'PlateD'}
    
    def _validate(self):
        folder_name = self.path.name
        m = self.name_regex.match(folder_name)
        if m is None:
            raise SubPlateWellImagingReplicateDirectoryNameError(folder_name)
            
        self.subplate = m.group('subplate')
        self.row = m.group('row')
        self.column = m.group('column')
        self.replicate = m.group('replicate')
        self.extra_info = m.group('extra_info')

    @property
    def name(self):
        return f'{self.subplate}{self.row}{self.column}_{self.replicate}'
        
    def _recurse(self):
        pass
    
    def __repr__(self):
        return f'<Well/replicate subplate {self.subplate}, row {self.row}, column {self.column}, replicate {self.replicate}>'
    
    def __str__(self):
        return repr(self)
    
    @property
    def _tif_filename(self):
        try:
            return next(iter(self.path.glob('*.tif')))
        except StopIteration:
            raise NoImagesFoundError(self.name)
            
    @property
    def _tifffile(self):
        return tifffile.TiffFile(self._tif_filename, 'rb')
            
    @property
    def image_metadata(self):
        with self._tifffile as tif:
            return xmltodict.parse(tif.ome_metadata, dict_constructor=dict)
    
    @property
    def phase_image(self):
        with self._tifffile as tif:
            return tif.series[0][0].asarray()

    @property
    def gene_image(self):
        with self._tifffile as tif:
            return tif.series[0][1].asarray()

    @property
    def dapi_image(self):
        with self._tifffile as tif:
            return tif.series[0][2].asarray()

    def _repr_png_(self):
        try:
            tif_fn = self._tif_filename
        except NoImagesFoundError:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        with self._tifffile as tif:
            for ax, p in zip(axes, tif.series[0]):
                ax.imshow(p.asarray(), cmap=plt.cm.Greys_r)
                ax.axis('off')
        f = io.BytesIO()
        fig.savefig(f, format="png", bbox_inches="tight")
        plt.close(fig)
        return f.getvalue()
