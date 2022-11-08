from pathlib import Path
import matplotlib.pyplot as plt
import io
from PIL import Image
import dataclasses
import base64
import pandas as pd
import itertools
import numpy as np
from typing import List

@dataclasses.dataclass
class SegmentedCell:
    length: float
    width: float
    area: float
    solidity: float
    flagellum_straight_length: float
    flagellum_path_length: float
    image: Image
    cell_body_coords: np.ndarray
    flagellum_coords: np.ndarray
    coordinates: tuple

    def _repr_html_(self):
        imgdata = base64.b64encode(self._repr_png_()).decode()
        if not np.isnan(self.flagellum_straight_length):
            flag_prop = f"""
            <li>Flag. length: {self.flagellum_path_length:.1f}µm</li>
            <li>Flag. straightness: {self.flagellum_straight_length/self.flagellum_path_length:.1f}</li>
            """
        else:
            flag_prop = ""

        return f'''
        <div style="padding: 5px; position: relative;">
          <img src="data:image/png;base64,{imgdata}">
          <ul style="position: absolute; bottom: 10px; left: 10px; list-style-type: none; background: white; opacity: 0.5; padding: 3px;">
            <li>Length: {self.length:.1f}µm</li>
            <li>Width: {self.width:.1f}µm</li>
            <li>Area: {self.area:.1f}µm²</li>
            <li>Solidity: {self.solidity:.2f}</li>
            {flag_prop}
          </ul>
        </div>'''
    
    def _repr_png_(self):
        return self.image._repr_png_()
    
    @property
    def bbox(self):
        cell_min = self.cell_body_coords.min(axis=0)
        cell_max = self.cell_body_coords.max(axis=0)
#        if self.flagellum_coords.shape[0] > 0:
#            flag_min = self.flagellum_coords.min(axis=0)
#            flag_max = self.flagellum_coords.max(axis=0)

#            minc = np.min([cell_min, flag_min], axis=0)
#            maxc = np.max([cell_max, flag_max], axis=0)
#        else:
        minc = cell_min
        maxc = cell_max
        
        return [minc[0], maxc[0], minc[1], maxc[1]]

@dataclasses.dataclass
class SegmentedCellCollection:
    name: str
    image_path: Path
    cell_list: List[SegmentedCell] = dataclasses.field(default_factory=list)
    
    def _repr_html_(self):
        return ('<div style="border: 1px solid black; display: flex; flex-wrap: wrap; justify-content: space-evenly;">' + 
                "".join([c._repr_html_() for c in self.cell_list]) +
                '</div>')
    
    def to_dataframe(self):
        return pd.DataFrame(self.cell_list).assign(name=self.name, image_path=self.image_path)
        
@dataclasses.dataclass
class SegmentedCellCollectionSet:
    sets: List[SegmentedCellCollection] = dataclasses.field(default_factory=list)
    
    def scatterplot(self):
        markers = itertools.cycle(['o', '<', '>', '^', 'v'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
        scs = []
        for coll in self.sets:
            widths = [c.width for c in coll.cell_list]
            lengths = [c.length for c in coll.cell_list]
            areas = [c.area for c in coll.cell_list]
            flag_lengths = [c.flagellum_path_length for c in coll.cell_list]
            scs.append(ax.scatter(lengths, widths, c=flag_lengths, s=20, marker=next(markers), #vmin=15, vmax=45,
                                  label=coll.name, edgecolors="black", plotnonfinite=True))
        plt.colorbar(scs[-1]).set_label("Flagellum length [µm]")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize="small", frameon=True)
        ax.set_xlabel("Length [µm]")
        ax.set_ylabel("Width [µm]")
        return fig
    
    def _repr_png_(self):
        fig = self.scatterplot()
        f = io.BytesIO()
        fig.savefig(f, format="png", bbox_inches="tight")
        plt.close()
        return f.getvalue()
    
    def to_dataframe(self):
        return pd.concat([coll.to_dataframe() for coll in self.sets])
    
    def to_hdf(self, path):
        df = self.to_dataframe()
        df.image = [np.array(image) for image in df.image]
        df.to_hdf(path, "SegmentedCellCollectionSet")
    
    @staticmethod
    def load_hdf(path):
        df = pd.read_hdf(path, "SegmentedCellCollectionSet")
        colls = []
        df.image = [Image.fromarray(image) for image in df.image]
        for (name, imgpath), g in df.groupby(["name", "image_path"]):
            cells = []
            for r in g.to_dict('records'):
                del r["image_path"]
                del r["name"]
                cells.append(SegmentedCell(**r))
            colls.append(SegmentedCellCollection(name=name, image_path=imgpath, cell_list=cells))
        return SegmentedCellCollectionSet(colls)
